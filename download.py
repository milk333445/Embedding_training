from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
import os

model_name = "moka-ai/m3e-base"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.save_pretrained("./models/m3e-base")
tokenizer.save_pretrained("./models/m3e-base")
