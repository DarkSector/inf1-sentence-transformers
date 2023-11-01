import os
import torch
import torch.neuron
from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

batch_size = 50

# this model has been traced with input size of 20 sentences <=256 len, max_length=256
model_dir = f"/home/ubuntu/distilbert-base-uncased-finetuned-sst-2-english/tmp{batch_size}"
model_name = f"neuron_model_{batch_size}.pt"

tokenizer = AutoTokenizer.from_pretrained(model_dir)
model_config = AutoConfig.from_pretrained(model_dir)

model = torch.jit.load(os.path.join(model_dir, model_name))

single_static_batch = ["Here's a simple sentence that we're going to produce embeddings for. We're going to replicate it batch_size times"] * batch_size

# This is just to keep this script running
mega_batch = [singe_static_batch] * 3000

for idx, batch in enumerate(mega_batch):
  
    embeddings = tokenizer(
        batches,
        return_tensors="pt",
        max_length=model_config.traced_sequence_length,
        padding="max_length",
        truncation=True,
    )
    neuron_inputs = tuple(embeddings.values())

    with torch.no_grad():
        predictions = model(*neuron_inputs)[0]
        scores = torch.nn.Softmax(dim=1)(predictions)
        # use mean_pooling and torch Functional here like we do in SageMaker
