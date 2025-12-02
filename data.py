from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import os
from PIL import Image
import io
import json
from llama_index.core.schema import TextNode, ImageNode
load_dotenv()

def data_loading(name, split, streaming, size, analysis=False):

    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Login Successful")

    dataset = load_dataset(name, split=split, streaming=streaming)
    # Turing only: shards = dataset.shard(num_shards=64, index=0)

    dataset_100 = dataset.take(size)

    samples = []
    for sample in tqdm(dataset_100, total=size, desc="Loading samples"):
        samples.append(sample)

    print(f"Loaded {len(samples)} samples")

    if analysis:
        sample = next(iter(samples[0]))
        print(sample)
        print(sample.keys())

    return samples

def data_clean(list):
  data = [element for element in list if element is not None]
  return data

def create_text_nodes():
    pass

def create_image_nodes():
    pass