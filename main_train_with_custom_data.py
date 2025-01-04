import tensorflow as tf
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import Dataset

# Step 1: Prepare the Dataset
data = [
    {"input": "Who made you?", "output": "I was made by Kirilov"},
]

# Create Dataset
dataset = Dataset.from_list(data)

# Step 2: Preprocess the Data
def preprocess(data):
    inputs = tokenizer(
        data["input"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf"
    )
    outputs = tokenizer(
        data["output"],
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="tf"
    )
    return {
        "input_ids": inputs["input_ids"][0],
        "attention_mask": inputs["attention_mask"][0],
        "labels": outputs["input_ids"][0],
    }

# Load tokenizer
model_name = "gpt2"  # You can replace this with a different model

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Add a padding token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Tokenize and preprocess the dataset
print("Preprocessing dataset...")
tokenized_data = [preprocess(example) for example in dataset]

# Convert to TensorFlow dataset
tf_dataset = tf.data.Dataset.from_generator(
    lambda: tokenized_data,
    output_signature={
        "input_ids": tf.TensorSpec(shape=(128,), dtype=tf.int32),
        "attention_mask": tf.TensorSpec(shape=(128,), dtype=tf.int32),
        "labels": tf.TensorSpec(shape=(128,), dtype=tf.int32),
    }
)

tf_dataset = tf_dataset.batch(4).shuffle(len(tokenized_data))

# Step 3: Load the Model
print("Loading model...")
model = TFAutoModelForCausalLM.from_pretrained(model_name)

# Resize model embeddings if new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss)

# Step 4: Train the Model
print("Starting training...")
model.fit(tf_dataset, epochs=1000)

# Step 5: Save the Fine-Tuned Model
print("Saving the model...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

# Step 6: Test the Fine-Tuned Model
from transformers import pipeline

print("Testing the model...")
conversation = pipeline("text-generation", model="./fine_tuned_model", tokenizer=tokenizer)
response = conversation("Who made you?", max_length=50, num_return_sequences=1)
print(response[0]["generated_text"])
