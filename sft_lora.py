from datasets import load_dataset

from transformers import (
    AutoTokenizer
)

from peft import (
    AutoPeftModelForCausalLM
)

from trl import (
    SFTTrainer,
    SFTConfig
)

import random
import numpy as np
import torch

# TRL patches the model with a Triton "chunked log-prob" LM head. Triton ships no
# Windows wheels, so this installs a pure-PyTorch implementation of the same
# kernel (identical results and gradients, one vocabulary chunk at a time).
import trl_windows_compat

trl_windows_compat.install()

SEED = 48391

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


MODEL_PATH = (
    "models/domain_adapter"
)


## Load domain-adapted model

model = (
    AutoPeftModelForCausalLM
    .from_pretrained(
        MODEL_PATH,
        is_trainable=True
    )
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)


#3 Load SFT dataset

dataset = load_dataset(
    "json",
    data_files=
        "data/sft_clean.jsonl",
    split="train"
)


dataset = dataset.train_test_split(
    test_size=0.1,
    seed=SEED
)



## Convert to prompt/completion format

def format_example(example):

    return {
        "prompt": (
            "You are a university IT "
            "support assistant.\n\n"
            "User: "
            + example["prompt"]
            + "\n\nAssistant:"
        ),

        "completion":
            " " + example["completion"]
    }


dataset = dataset.map(
    format_example
)



## Training configuration


config = SFTConfig(
    output_dir=
        "models/specialized_adapter",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    learning_rate=1e-4,
    logging_steps=20,
    eval_strategy="epoch",
    save_strategy="epoch",
    max_length=256,
    completion_only_loss=True,
    use_liger_kernel=False,
    report_to="none",
    seed=SEED
)



# Trainer


trainer = SFTTrainer(
    model=model,
    args=config,
    train_dataset=
        dataset["train"],
    eval_dataset=
        dataset["test"],
    processing_class=tokenizer
)



# Fine-tune

trainer.train()



# Save

model.save_pretrained(
    "models/specialized_adapter"
)

tokenizer.save_pretrained(
    "models/specialized_adapter"
)