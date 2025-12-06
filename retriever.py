import chromadb
from llama_index.core import StorageContext
from chromadb.utils.data_loaders import ImageLoader
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.indices import MultiModalVectorStoreIndex
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.clip import ClipEmbedding
from llama_index.core import Settings

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
    return storage_context

def embeddings(storage_context, nodes):
    """
        Splitting nodes into batches to avoid exceeding ChromaDB's batch size limit of 5461
    """
    BATCH_SIZE = 5000

    Settings.embed_model = OllamaEmbedding(
        model_name="nomic-embed-text",
        base_url="http://localhost:11434",
        ollama_additional_kwargs={
            "num_ctx": 512,
        },
    )

    index = None

    for i in range(0, len(nodes), BATCH_SIZE):
        batch_nodes = nodes[i:i + BATCH_SIZE]

        if index is None:
            index = MultiModalVectorStoreIndex(
                nodes=batch_nodes,
                storage_context=storage_context,
                image_embed_model=ClipEmbedding(),
            )
        else:
            index.insert_nodes(batch_nodes)

    print("Index Created")

    return index