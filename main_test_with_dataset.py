import os

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.text import Text

import warnings

# Suppress warnings in this block
with warnings.catch_warnings():
    pass

# Initialize rich console
console = Console()

# Step 1: Load Dataset
console.log("Loading dataset...", style="bold blue")
dataset = load_dataset("AlekseyKorshuk/persona-chat", split="train")  # Use train split
data = [{"input": example["utterances"][0]["history"][-1], "output": example["utterances"][0]["candidates"][0]} for example in dataset]
dataset = Dataset.from_list(data)

# Step 2: Preprocess the Data
console.log("Preprocessing data...", style="bold green")


def preprocess(data):
    inputs = tokenizer(data["input"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    outputs = tokenizer(data["output"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": outputs["input_ids"].squeeze()
    }


# Step 3: Load Tokenizer and Model
model_name = "gpt2"
console.log(f"Loading tokenizer and model: {model_name}...", style="bold yellow")
tokenizer = AutoTokenizer.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

model = AutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
console.log("Tokenizing dataset...", style="bold cyan")
tokenized_dataset = dataset.map(preprocess, batched=False)

os.system("cls")

# Step 4: Define Training Arguments
console.log("Setting up training arguments...", style="bold magenta")
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    save_steps=20,
    save_total_limit=1,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Step 5: Define Trainer
console.log("Initializing Trainer...", style="bold white")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Step 6: Train the Model
console.log("Starting training...", style="bold green")
trainer.train()

# Step 7: Save the Fine-Tuned Model
console.log("Saving the fine-tuned model...", style="bold yellow")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 8: Test the Fine-Tuned Model
from transformers import pipeline

console.log("Testing the fine-tuned model...", style="bold cyan")
conversation = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer)
response = conversation("Hi, how are you?", max_length=50, num_return_sequences=1)

# Display the response with rich styling
console.log("Model Response:", style="bold magenta")
console.print(Text(response[0]["generated_text"], style="bold green"))
