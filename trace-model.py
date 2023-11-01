import os
import torch.neuron
from transformers import AutoTokenizer, AutoModel
import argparse
import pathlib

parser = argparse.ArgumentParser(description='Build for inf1')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--model_id', type=str, default="sentence-transformers/all-MiniLM-L6-v2")

args = parser.parse_args()

batch_size = args.batch_size
model_id = args.model_id

print(f"Starting to compile model with batch size {batch_size}... and model_id {model_id}\n")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id, torchscript=True)

# create dummy input for max length 128
_dummy_input = ["The movie had stunning visuals and a unique storyline, but the pacing felt off and some scenes were confusing"]
dummy_input = batch_size * _dummy_input

max_length = 256
encoded_input = tokenizer(dummy_input, max_length=max_length, padding='max_length', truncation=True, return_tensors='pt')
traced_model = torch.neuron.trace(model, encoded_input['input_ids'], strict=False)

directory = f"models/{model_id}"
pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

print(f"Done compiling sending output to {directory}/tmp{batch_size} ...\n")

# # save tokenizer, neuron model and config for later use
save_dir=f"{directory}/tmp{batch_size}"
os.makedirs(save_dir, exist_ok=True)
traced_model.save(os.path.join(save_dir,f"neuron_model_{batch_size}.pt"))
tokenizer.save_pretrained(save_dir)
