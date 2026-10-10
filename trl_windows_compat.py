"""Windows-compatible replacement for TRL's Triton chunked-log-prob kernel.

Why this file exists
--------------------
`trl` 1.15 always calls `add_fused_lm_head()` when an `SFTTrainer` is built
(`trl/trainer/sft_trainer.py`). That patches `model.forward` so that training
scores the LM head through `trl.kernels.ChunkedLogProbFunction`.

`trl.kernels` imports `triton`, and Triton publishes no `win_amd64` wheels, so on
Windows the import fails, TRL falls back to `_ChunkedLogProbFunction = None`, and
training dies with::

    AttributeError: 'NoneType' object has no attribute 'apply'

(`liger-kernel`, the other package that would supply this, is unusable on Windows
for the same reason: it depends on `triton`.)

What this does
--------------
It provides a pure-PyTorch `ChunkedLogProbFunction` with the same signature,
return contract and gradients as the Triton implementation, and installs it into
`trl.trainer.utils` so TRL's fused LM head works unchanged.

The projection is still evaluated one vocabulary chunk at a time, so the full
`[tokens, vocab]` logits tensor is never materialised — the memory advantage of
the original is preserved. The maths is identical to a standard
cross-entropy over the transformed logits:

    z = logits * logit_scale
    z = softcap * tanh(z / softcap)      # only when the config sets a softcap
    z = z / temperature

    log p_target = z_target - logsumexp(z)
    entropy      = -sum_v p_v * log p_v
"""

from __future__ import annotations

import torch

OPTIONAL_OUTPUTS = ("entropy", "log_sum_sq_probs", "mean_logits", "is_top1")

# Vocabulary width of one projection tile.
DEFAULT_CHUNK_SIZE = 8192


def _transform(y: torch.Tensor, logit_scale: float, softcap: float | None, inv_t: float) -> torch.Tensor:
    """Apply the LM-head scaling pipeline the kernels use, in float32."""
    z = y * logit_scale
    if softcap is not None:
        z = softcap * torch.tanh(z / softcap)
    return z * inv_t


class ChunkedLogProbFunction(torch.autograd.Function):
    """Per-token log-probabilities of ``hidden @ weight.T`` without full logits."""

    @staticmethod
    def forward(
        ctx,
        hidden: torch.Tensor,  # [N, H]
        weight: torch.Tensor,  # [V, H]
        bias: torch.Tensor | None,  # [V]
        targets: torch.Tensor,  # [N]
        temperature: float,
        chunk_size: int,
        final_logit_softcapping: float | None = None,
        logit_scale: float = 1.0,
        outputs: tuple[str, ...] = OPTIONAL_OUTPUTS,
    ):
        ctx.set_materialize_grads(False)
        hidden = hidden.contiguous()
        weight = weight.contiguous()

        device = hidden.device
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]
        if chunk_size is None or chunk_size <= 0:
            chunk_size = DEFAULT_CHUNK_SIZE

        compute_dtype = hidden.dtype
        if torch.is_autocast_enabled(device.type):
            try:
                compute_dtype = torch.get_autocast_dtype(device.type)
            except Exception:  # pragma: no cover - very old torch
                pass

        inv_t = 1.0 / temperature
        softcap = final_logit_softcapping

        want_entropy = "entropy" in outputs
        want_log_sum_sq = "log_sum_sq_probs" in outputs
        want_mean_logits = "mean_logits" in outputs
        want_is_top1 = "is_top1" in outputs

        running_max = torch.full((n_tokens,), float("-inf"), device=device, dtype=torch.float32)
        sum_exp = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
        x_sum_exp = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
        sq_sum_exp = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
        z_sum = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
        target_logit = torch.zeros((n_tokens,), device=device, dtype=torch.float32)
        if want_is_top1:
            max_before_target = torch.full((n_tokens,), float("-inf"), device=device, dtype=torch.float32)

        hidden_c = hidden.to(compute_dtype)

        with torch.no_grad():
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                y = hidden_c @ weight[start:end].to(compute_dtype).t()
                if bias is not None:
                    y = y + bias[start:end].to(compute_dtype)
                z = _transform(y.float(), logit_scale, softcap, inv_t)

                chunk_max = z.amax(dim=1)
                new_max = torch.maximum(running_max, chunk_max)
                rescale = torch.exp(running_max - new_max)
                e = torch.exp(z - new_max.unsqueeze(1))

                if want_entropy:
                    x_sum_exp = x_sum_exp * rescale + (e * z).sum(dim=1)
                if want_log_sum_sq:
                    sq_sum_exp = sq_sum_exp * rescale * rescale + (e * e).sum(dim=1)
                if want_mean_logits:
                    z_sum = z_sum + z.sum(dim=1)
                if want_is_top1:
                    # Whole chunk lies before the target?
                    strictly_before = targets >= end
                    if strictly_before.any():
                        max_before_target[strictly_before] = torch.maximum(
                            max_before_target[strictly_before], chunk_max[strictly_before]
                        )
                    in_chunk_local = (targets >= start) & (targets < end)
                    if in_chunk_local.any():
                        local = targets[in_chunk_local] - start
                        positions = torch.arange(end - start, device=device)
                        before_mask = positions.unsqueeze(0) < local.unsqueeze(1)
                        masked = torch.where(
                            before_mask,
                            z[in_chunk_local],
                            torch.full((), float("-inf"), device=device, dtype=torch.float32),
                        )
                        max_before_target[in_chunk_local] = torch.maximum(
                            max_before_target[in_chunk_local], masked.amax(dim=1)
                        )

                sum_exp = sum_exp * rescale + e.sum(dim=1)
                running_max = new_max

                # Gather the target logit from whichever chunk holds it.
                in_chunk = (targets >= start) & (targets < end)
                if in_chunk.any():
                    rows = torch.nonzero(in_chunk, as_tuple=False).squeeze(1)
                    target_logit[rows] = z[rows, targets[rows] - start]

        log_z = running_max + torch.log(sum_exp)
        logprobs = target_logit - log_z
        entropy = log_z - x_sum_exp / sum_exp if want_entropy else None
        log_sum_sq_probs = torch.log(sq_sum_exp) - 2 * torch.log(sum_exp) if want_log_sum_sq else None
        mean_logits = z_sum / vocab if want_mean_logits else None
        is_top1 = (target_logit >= running_max) & (target_logit > max_before_target) if want_is_top1 else None

        # `log_z` and `entropy` are both needed in backward; reuse the slot when
        # entropy was not requested, exactly like the reference implementation.
        ctx.save_for_backward(
            hidden,
            weight,
            bias if bias is not None else torch.empty(0, device=device, dtype=compute_dtype),
            targets,
            log_z,
            log_z if entropy is None else entropy,
        )
        ctx.has_bias = bias is not None
        ctx.compute_dtype = compute_dtype
        ctx.chunk_size = chunk_size
        ctx.inv_t = inv_t
        ctx.softcap = softcap
        ctx.logit_scale = logit_scale
        ctx.mark_non_differentiable(
            *(x for x in (log_sum_sq_probs, mean_logits, is_top1) if x is not None)
        )

        return logprobs, entropy, log_sum_sq_probs, mean_logits, is_top1

    @staticmethod
    def backward(ctx, grad_logprobs, grad_entropy, *_):
        hidden, weight, bias_saved, targets, log_z, entropy = ctx.saved_tensors
        bias = bias_saved if ctx.has_bias else None
        compute_dtype = ctx.compute_dtype
        chunk_size = ctx.chunk_size
        inv_t = ctx.inv_t
        softcap = ctx.softcap
        logit_scale = ctx.logit_scale

        needs_hidden, needs_weight, needs_bias = ctx.needs_input_grad[:3]
        device = hidden.device
        n_tokens = hidden.shape[0]
        vocab = weight.shape[0]

        grad_hidden = (
            torch.zeros(hidden.shape, device=device, dtype=torch.float32) if needs_hidden else None
        )
        grad_weight = (
            torch.zeros(weight.shape, device=device, dtype=torch.float32) if needs_weight else None
        )
        grad_bias = (
            torch.zeros((vocab,), device=device, dtype=torch.float32) if (needs_bias and bias is not None) else None
        )

        gl = grad_logprobs.float().contiguous() if grad_logprobs is not None else None
        ge = grad_entropy.float().contiguous() if grad_entropy is not None else None

        hidden_c = hidden.to(compute_dtype)

        with torch.no_grad():
            for start in range(0, vocab, chunk_size):
                end = min(start + chunk_size, vocab)
                w_chunk = weight[start:end].to(compute_dtype)
                y = hidden_c @ w_chunk.t()
                if bias is not None:
                    y = y + bias[start:end].to(compute_dtype)

                y32 = y.float()
                z = _transform(y32, logit_scale, softcap, inv_t)
                log_p = z - log_z.unsqueeze(1)
                p = torch.exp(log_p)

                grad = torch.zeros_like(z)
                if gl is not None:
                    # d(log p_target) / dz_j = 1[j = target] - p_j
                    positions = torch.arange(start, end, device=device)
                    one_hot = (positions.unsqueeze(0) == targets.unsqueeze(1)).to(z.dtype)
                    grad = grad + gl.unsqueeze(1) * (one_hot - p)
                if ge is not None:
                    # d(entropy) / dz_j = -p_j * (log p_j + entropy)
                    grad = grad - ge.unsqueeze(1) * p * (log_p + entropy.unsqueeze(1))

                # Chain rule back through z = softcap * tanh(y/softcap) * inv_t * logit_scale
                grad = grad * inv_t
                if softcap is not None:
                    t = torch.tanh(y32 / softcap)
                    grad = grad * (1.0 - t * t)
                grad = grad * logit_scale

                if grad_hidden is not None:
                    grad_hidden += grad.to(w_chunk.dtype) @ w_chunk
                if grad_weight is not None:
                    grad_weight[start:end] += grad.t() @ hidden_c
                if grad_bias is not None:
                    grad_bias[start:end] += grad.sum(dim=0)

        return (
            grad_hidden.to(hidden.dtype) if grad_hidden is not None else None,
            grad_weight.to(weight.dtype) if grad_weight is not None else None,
            grad_bias.to(bias.dtype) if (grad_bias is not None and bias is not None) else None,
            None,  # targets
            None,  # temperature
            None,  # chunk_size
            None,  # final_logit_softcapping
            None,  # logit_scale
            None,  # outputs
        )


def install() -> bool:
    """Install this implementation into TRL when Triton's is unavailable.

    Returns True when TRL's fused LM head can be used.
    """
    import trl.trainer.utils as trl_utils

    if trl_utils._ChunkedLogProbFunction is None:  # noqa: SLF001 - public replacement
        trl_utils._ChunkedLogProbFunction = ChunkedLogProbFunction  # noqa: SLF001
        return True
    return True


if __name__ == "__main__":  # pragma: no cover - self-check
    import trl.trainer.utils as trl_utils

    install()
    print("Installed ChunkedLogProbFunction:", trl_utils._ChunkedLogProbFunction is ChunkedLogProbFunction)

    torch.manual_seed(0)
    hidden = torch.randn(6, 8, dtype=torch.float32, requires_grad=True)
    weight = torch.randn(32, 8, dtype=torch.float32, requires_grad=True)
    bias = torch.randn(32, dtype=torch.float32, requires_grad=True)
    targets = torch.randint(0, 32, (6,))

    got = ChunkedLogProbFunction.apply(
        hidden, weight, bias, targets, 1.0, 8, None, 1.0, ("log_probs", "entropy", "is_top1")
    )
    logits = hidden @ weight.t() + bias
    expected = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
    print("logprob max abs diff:", float((got[0] - expected).abs().max()))

    probs = torch.softmax(logits, dim=-1)
    expected_entropy = -(probs * torch.log(probs)).sum(dim=1)
    print("entropy max abs diff:", float((got[1] - expected_entropy).abs().max()))
    print("is_top1 match:", bool((got[4] == (logits.argmax(dim=1) == targets)).all()))

    got[0].sum().backward()
    ref_logits = hidden.detach().clone().requires_grad_(True)
    ref_w = weight.detach().clone().requires_grad_(True)
    ref_b = bias.detach().clone().requires_grad_(True)
    ref = torch.log_softmax(ref_logits @ ref_w.t() + ref_b, dim=-1).gather(1, targets[:, None]).squeeze(1)
    ref.sum().backward()
    print("grad hidden max abs diff:", float((hidden.grad - ref_logits.grad).abs().max()))
    print("grad weight max abs diff:", float((weight.grad - ref_w.grad).abs().max()))
    print("grad bias   max abs diff:", float((bias.grad - ref_b.grad).abs().max()))
