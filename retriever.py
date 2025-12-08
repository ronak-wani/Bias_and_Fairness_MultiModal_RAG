import chromadb
from llama_index.core import StorageContext
from chromadb.utils.data_loaders import ImageLoader
from llama_index.core.schema import TextNode, ImageNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core import Settings
from tqdm import tqdm
from data import data_loading, create_nodes

"""
Using two different vector stores for images and text
because CLIP Model is limited to only 76 tokens
and truncating text will lead to loss of information
"""

def multimodal_vector_db():

    chroma_client = chromadb.PersistentClient()

    chroma_image_collection = chroma_client.get_or_create_collection(
        "multimodal-rag-images",
        data_loader=ImageLoader(),
    )

    chroma_text_collection = chroma_client.get_or_create_collection(
        "multimodal-rag-text",
    )

    image_vector_store = ChromaVectorStore(chroma_collection=chroma_image_collection)
    text_vector_store = ChromaVectorStore(chroma_collection=chroma_text_collection)

    storage_context = StorageContext.from_defaults(
        vector_store=text_vector_store,
        image_store=image_vector_store
    )
    print("Storage Context Created")

    text_count = chroma_text_collection.count()
    image_count = chroma_image_collection.count()

    text_embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={
            "num_ctx": 512,
        },
    )

    Settings.embed_model = text_embed_model

    if text_count == 0 and image_count == 0:
        corpus = data_loading("HuggingFaceM4/OBELICS", "train", True, 50000, False)
        nodes = create_nodes(corpus)
        print(f"Total nodes created: {len(nodes)}")
        print(f"Text nodes: {sum(1 for n in nodes if isinstance(n, TextNode))}")
        print(f"Image nodes: {sum(1 for n in nodes if isinstance(n, ImageNode))}")

        """
        Splitting nodes into batches to avoid exceeding ChromaDB's batch size limit of 5461
        """

        batch_size = 1000
        total_batches = (len(nodes) + batch_size - 1) // batch_size
        index = None

        for i in tqdm(range(0, len(nodes), batch_size),
                      total=total_batches,
                      desc="Creating index batches",
                      unit="batch"):
            batch_nodes = nodes[i:i + batch_size]

            if index is None:
                index = MultiModalVectorStoreIndex(
                    nodes=batch_nodes,
                    storage_context=storage_context,
                    embed_model=text_embed_model,
                    image_embed_model=ClipEmbedding(),
                )
            else:
                index.insert_nodes(batch_nodes)

        print("Index Created")

    else:
        index = MultiModalVectorStoreIndex.from_vector_store(
            text_vector_store=text_vector_store,
            image_vector_store=image_vector_store,
            embed_model=text_embed_model,
            image_embed_model=ClipEmbedding(),
        )
        print("Existing index loaded")

    return index