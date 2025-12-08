import asyncio
import aiohttp
from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login
from tqdm import tqdm
import os
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
    text_nodes = []
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
        text_nodes.append(text_node)

        image_nodes = process_images_sync(images, metadata)
        nodes = text_nodes + image_nodes

    return nodes


async def fetch_and_process_image(session, img_url, metadata):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Referer': 'https://www.google.com/'
    }

    try:
        async with session.get(img_url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
            response.raise_for_status()
            content = await response.read()

            loop = asyncio.get_event_loop()
            image = await loop.run_in_executor(
                None,
                lambda: Image.open(BytesIO(content)).convert("RGB")
            )
            image_base64 = await loop.run_in_executor(None, image_to_base64, image)

            image_node = ImageNode(
                image=image_base64,
                image_url=img_url,
                image_mimetype="JPEG",
                metadata=metadata
            )
            print("Success: " + img_url)
            return image_node

    except Exception as e:
        print(f"✗ Error processing image: {img_url}")
        print(f"  Error Type: {type(e).__name__}")
        print(f"  Error Message: {str(e)}")
        return None


async def process_images_async(images, metadata, max_concurrent=100):
    if not images:
        return []

    image_urls = data_clean(images)
    print(image_urls)

    connector = aiohttp.TCPConnector(limit=max_concurrent, limit_per_host=5)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            fetch_and_process_image(session, img_url, metadata)
            for img_url in image_urls
        ]

        results = await asyncio.gather(*tasks, return_exceptions=False)

        nodes = [node for node in results if node is not None]

    return nodes

def process_images_sync(images, metadata):
    """Synchronous wrapper for async image processing"""
    return asyncio.run(process_images_async(images, metadata))
