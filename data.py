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

    sub_sampled_dataset = dataset.take(size)

    samples = []
    for sample in tqdm(sub_sampled_dataset, total=size, desc="Loading samples"):
        samples.append(sample)

    print(f"Loaded {len(samples)} samples")

    if analysis:
        print(samples[0])
        print(samples[0].keys())

    return samples

def data_clean(list):
  data = [element for element in list if element is not None]
  return data

def image_to_base64(image):
    copy = io.BytesIO()
    image.save(copy, format="JPEG")
    image_base64 = base64.b64encode(copy.getvalue()).decode()
    return image_base64

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

        text_node = TextNode(text=content, metadata=metadata)
        nodes.append(text_node)

        if images:
            image_urls = data_clean(images)
            print(image_urls)

            for idx, img_url in enumerate(image_urls):
                try:
                    image_res = requests.get(img_url, timeout=10)
                    image_res.raise_for_status()  # Raise exception for bad status codes
                    image = Image.open(BytesIO(image_res.content)).convert("RGB")
                    image_base64 = image_to_base64(image)
                    image_node = ImageNode(image=image_base64, image_url=img_url, image_mimetype="JPEG",
                                           metadata=metadata)
                    nodes.append(image_node)
                except Exception as e:
                    print(f"✗ Error processing image: {img_url}")
                    print(f"  Error Type: {type(e).__name__}")
                    print(f"  Error Message: {str(e)}")
    return nodes