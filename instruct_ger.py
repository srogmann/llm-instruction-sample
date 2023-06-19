from datasets import Dataset
import evaluate
import numpy as np
import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
from typing import Any, Dict, List, Tuple, Union

input_file = "instruct_ger_bsp_1.txt"
# model_name = "malteos/bloom-350m-german"
# model_name = "malteos/gpt2-wechsel-german-ds-meg"
model_name = "malteos/bloom-1b5-clp-german"
output_dir = "instruct_ger"
output_model = "instruct_ger_bsp_1_1b5_ep4"

# See https://docs.wandb.ai/guides/integrations/huggingface
os.environ["WANDB_DISABLED"] = "True"

# Instruction and expected answer are separated by '#'.
pInstruct = re.compile("(.*[^ ]) *# *(.*)")
questions = []
answers = []
sentences = []
idxEval = None
comment_char = "#"
with open(input_file) as file:
    for line in file:
        s = line.rstrip()
        if s.startswith(comment_char) and "_EVAL_" in s:
            idxEval = len(sentences)
            continue
        if len(s) == 0 or s[0] == comment_char:
            # Ignore empty lines and comments.
            continue
        m = pInstruct.match(s)
        if m:
            q = f"Es folgt eine Anweisung, die eine Aufgabe beschreibt. Schreibe eine passende Antwort. Anweisung: {m.group(1)} Antwort: \n"
            questions.append(q)
            a = m.group(2)
            answers.append(a)
            sentences.append(q + a)
            if len(sentences) == 1:
                print(f"First training sentence: {sentences[0]}")
            if idxEval is not None and len(sentences) == idxEval + 1:
                print(f"First evaluation sentence: {sentences[idxEval]}")

if idxEval == None:
    print("No evaluation-marker _EVAL_ found. Use training-data as evaluation-data.")
    train = sentences
    eval = sentences
else:
    train = sentences[0:idxEval]
    eval = sentences[idxEval:]

print(f"#train: {len(train)}")
print(f"#eval: {len(eval)}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
print(f"Model: {model_name}")

tokenizer.pad_token = tokenizer.eos_token
encoded_train = tokenizer(train, padding=True, truncation=True)
encoded_eval = tokenizer(eval, padding=True, truncation=True)


def transpose_enc(enc):
    lm = []
    for i in range(len(enc["input_ids"])):
        ids = enc["input_ids"][i]
        att = enc["attention_mask"][i]
        lm.append({"input_ids": ids, "attention_mask": att, "labels": ids})
    return lm


lm_train = transpose_enc(encoded_train)
lm_eval = transpose_enc(encoded_eval)
print(f"#inputs: {len(encoded_train)}")

model = AutoModelForCausalLM.from_pretrained(model_name)
metric = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# We display the token of linefeed.
response_token_ids = tokenizer("\n")[0].ids
print(f"response_token_ids: {response_token_ids}")


# See https://github.com/databrickslabs/dolly/blob/master/training/trainer.py
#
class DataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):
    def torch_call(
        self, examples: List[Union[List[int], Any, Dict[str, Any]]]
    ) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        labels = batch["labels"].clone()

        for i in range(len(examples)):
            response_token_ids_start_idx = None
            b = batch["labels"][i]
            for idx in range(len(b)):
                if b[idx] == response_token_ids[0]:
                    response_token_ids_start_idx = idx
                    break
            if response_token_ids_start_idx is None:
                raise RuntimeError(
                    f'Index {i} of {len(examples)}: Could not find response key {response_token_ids[0]} of {response_token_ids} in token IDs {batch["labels"][i]}'
                )

            response_token_ids_end_idx = response_token_ids_start_idx + 1

            # Make pytorch loss function ignore all tokens up through the end of the response key
            labels[i, :response_token_ids_end_idx] = -100

        batch["labels"] = labels

        return batch


data_collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)

training_args = TrainingArguments(
    output_dir=output_dir,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    num_train_epochs=4,
    no_cuda=True,
    learning_rate=2e-5,
    weight_decay=0.01,
    push_to_hub=False,
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_train,
    eval_dataset=lm_eval,
    tokenizer=tokenizer,
    data_collator=data_collator,
)
print("Start training ...")

trainer.train(resume_from_checkpoint=False)
trainer.save_model(f"{output_dir}/{output_model}")
print(f"Training finished. Wrote model to {output_dir}/{output_model}")
