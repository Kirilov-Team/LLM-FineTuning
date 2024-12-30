from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from datasets import Dataset

# Step 1: Prepare the Dataset
data = [

    {"input": "Who made you?", "output": "I was made by Kirilov"},
]

# Create Dataset
dataset = Dataset.from_list(data)

# Step 2: Preprocess the Data
def preprocess(data):
    inputs = tokenizer(data["input"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    outputs = tokenizer(data["output"], truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    return {
        "input_ids": inputs["input_ids"].squeeze(),
        "attention_mask": inputs["attention_mask"].squeeze(),
        "labels": outputs["input_ids"].squeeze()
    }


# Load tokenizer
model_name = "gpt2"  # You can replace this with a different model

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Add [PAD] as the padding token

# Load model
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(model_name)

# Resize model embeddings if new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Tokenize the dataset
print("Tokenizing dataset...")
tokenized_dataset = dataset.map(preprocess, batched=False)


# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=1000,
    per_device_train_batch_size=4,
    save_steps=10,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=5,
    evaluation_strategy="no",
    learning_rate=5e-5,
    weight_decay=0.01,
)

# Step 5: Define Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Step 6: Train the Model
print("Starting training...")
trainer.train()

# Step 7: Save the Fine-Tuned Model
print("Saving the model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 8: Test the Fine-Tuned Model
from transformers import pipeline

print("Testing the model...")
conversation = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer)
response = conversation("Who made you?", max_length=50, num_return_sequences=1)
print(response[0]["generated_text"])
