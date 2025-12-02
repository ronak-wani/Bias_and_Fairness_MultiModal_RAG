from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import os
import requests
from io import BytesIO
import base64
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
        print(samples[0])
        print(samples[0].keys())

    return samples

def data_clean(list):
  data = [element for element in list if element is not None]
  return data

def create_nodes(corpus):
    nodes = []

    for item in corpus:
        texts = item.get('texts')
        images = item.get('images')
        metadata = item.get('metadata')

        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        metadata = data_clean(metadata)
        metadata = metadata[0]

        content = data_clean(texts)
        content = "\n".join(content)

        text_node = TextNode(text=content, metadata=metadata[0])
        nodes.append(text_node)

        if images:
            image_url = data_clean(images)
            print(image_url)
            try:
                image_res = requests.get(image_url[0])
                image = Image.open(BytesIO(image_res.content)).convert("RGB")
                copy = io.BytesIO()
                image.save(copy, format="JPEG")
                image_base64 = base64.b64encode(copy.getvalue()).decode()
                image_node = ImageNode(image=image_base64, image_url=image_url[0], image_mimetype="JPEG",
                                       metadata=metadata[0])
                nodes.append(image_node)
            except:
                print(f"Error processing image")

    return nodes