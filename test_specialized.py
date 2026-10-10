import torch
import json
import sys

from transformers import (
    AutoTokenizer
)

from peft import (
    AutoPeftModelForCausalLM
)


# Which adapter to test. Defaults to the V1 adapter, and it can be pointed at the
# V2/V3 adapters without editing the file:
#
#   python test_specialized.py models/specialized_adapter_v2
#
MODEL_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "models/specialized_adapter"
)


model = (
    AutoPeftModelForCausalLM
    .from_pretrained(
        MODEL_PATH
    )
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_PATH
    )
)


model.eval()


with open(
    "data/sft_challenge_test.jsonl",
    "r",
    encoding="utf-8"
) as file:

    tests = [
        json.loads(line)
        for line in file
    ]


for test in tests:

    prompt = (
        "You are a university IT "
        "support assistant.\n\n"
        "User: "
        + test["prompt"]
        + "\n\nAssistant:"
    )


    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )


    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False
        )


    response = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )


    print("\n====================")
    print("QUESTION:")
    print(test["prompt"])

    print("\nEXPECTED POINTS:")
    print(test["expected_points"])

    print("\nMODEL:")
    print(response)