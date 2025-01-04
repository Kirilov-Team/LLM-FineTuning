import os
import warnings
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM
from datasets import Dataset, load_dataset
from rich.console import Console
from rich.text import Text

# Disable warnings
with warnings.catch_warnings():
    pass

# Start rich console
console = Console()

# Step 1: Load the dataset
console.log("Loading dataset...", style="bold blue")
dataset = load_dataset("AlekseyKorshuk/persona-chat", split="train")
data = [
    {"input": example["utterances"][0]["history"][-1], "output": example["utterances"][0]["candidates"][0]}
    for example in dataset
]
dataset = Dataset.from_list(data)

# Step 2: Preprocess the data
console.log("Preprocessing data...", style="bold green")

model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

def preprocess(data):
    inputs = tokenizer(data["input"], truncation=True, padding="max_length", max_length=128)
    outputs = tokenizer(data["output"], truncation=True, padding="max_length", max_length=128)
    return {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": outputs["input_ids"],
    }

# Tokenize the dataset
tokenized_data = dataset.map(preprocess, batched=False)
input_ids = tf.constant([example["input_ids"] for example in tokenized_data])
attention_mask = tf.constant([example["attention_mask"] for example in tokenized_data])
labels = tf.constant([example["labels"] for example in tokenized_data])

# Step 3: Prepare the TensorFlow dataset
console.log("Preparing TensorFlow dataset...", style="bold cyan")
train_dataset = tf.data.Dataset.from_tensor_slices(({"input_ids": input_ids, "attention_mask": attention_mask}, labels))
train_dataset = train_dataset.shuffle(len(train_dataset)).batch(4)

# Step 4: Load the model
console.log(f"Loading model: {model_name}...", style="bold yellow")
model = TFAutoModelForCausalLM.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Step 5: Compile the model
console.log("Compiling model...", style="bold magenta")
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, weight_decay=0.01)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss_fn)

# Step 6: Train the model
console.log("Starting training...", style="bold green")
model.fit(train_dataset, epochs=3)

# Step 7: Save the model
console.log("Saving the fine-tuned model...", style="bold yellow")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 8: Test the model
from transformers import pipeline

console.log("Testing the fine-tuned model...", style="bold cyan")
conversation = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer)
response = conversation("Hi, how are you?", max_length=50, num_return_sequences=1)

# Display the response with rich styling
console.log("Model Response:", style="bold magenta")
console.print(Text(response[0]["generated_text"], style="bold green"))
