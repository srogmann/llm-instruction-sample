import sys
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)

model_name = "instruct_ger/instruct_ger_bsp_1_1b5_ep4"
print(f"Load model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.half().cuda()


class StopOnTokens(StoppingCriteria):
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        stop_ids = [50278, 50279, 50277, 1, 0]
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False


print(f"Prompt:")
for myText in sys.stdin:
    prompt = f"Anweisung: {myText}"
    print(f"Command: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    tokens = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.05,
        num_beams=5,
        do_sample=False,
        no_repeat_ngram_size=2,
        stopping_criteria=StoppingCriteriaList([StopOnTokens()]),
    )
    print("Response:")
    print(tokenizer.decode(tokens[0], skip_special_tokens=True))
    print()
    print(f"Prompt:")
