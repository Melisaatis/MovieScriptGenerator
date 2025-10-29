# train_mistral_lora.py
import os
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"  # or mistralai/Mistral-7B-v0.3
TRAIN_FILE = "dataset/train.jsonl"
VALID_FILE = "dataset/valid.jsonl"
OUTPUT_DIR = "mistral-lora-output"
BATCH_SIZE = 1    # per device; use gradient_accumulation_steps to emulate bigger batch
EPOCHS = 3
LR = 2e-4
MAX_LENGTH = 1024  # token context window to use
DEVICE_MAP = "auto"

def load_jsonl_dataset(train_path, valid_path):
    ds = load_dataset("json", data_files={"train": train_path, "validation": valid_path})
    return ds

def tokenize_function(examples, tokenizer):
    prompts = examples["prompt"]
    completions = examples["completion"]
    inputs = []
    for p, c in zip(prompts, completions):
        # we format final training string = prompt + completion + stop token
        txt = p + c
        inputs.append(txt)
    tokenized = tokenizer(inputs, truncation=True, max_length=MAX_LENGTH)
    # labels = input_ids (causal LM)
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

def main():
    ds = load_jsonl_dataset(TRAIN_FILE, VALID_FILE)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    # if tokenizer has no pad_token set it:
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # load model in 4-bit (bitsandbytes) to save memory if supported
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        load_in_4bit=True,
        device_map=DEVICE_MAP,
        trust_remote_code=True,
    )

    # prepare for kbit training and apply LoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    # tokenization + dataset mapping
    tokenized = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=ds["train"].column_names)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,  # effective batch size = BATCH_SIZE * grad_accum * n_gpus
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,
        logging_steps=50,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        fp16=True,
        optim="paged_adamw_32bit",
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=data_collator,
    )

    trainer.train()
    # save peft adapters only
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Training done. LoRA adapters & tokenizer saved to", OUTPUT_DIR)

if __name__ == "__main__":
    main()
