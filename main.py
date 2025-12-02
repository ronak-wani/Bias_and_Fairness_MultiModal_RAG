from data import data_loading
from retriever import multimodal_vector_db


class MultiModalRAG:
    mllm: str
    def __init__(self, mllm):
        self.mllm = mllm

if __name__ == "__main__":
    corpus = data_loading("HuggingFaceM4/OBELICS", "train", True, 100)
    # nodes =
    storage_context = multimodal_vector_db()
    # = embeddings(storage_context, nodes)

    benchmark = data_loading("ucf-crcv/SB-Bench", "real", True)

    Text_Only_Retrieval = MultiModalRAG("gemma3")
    Image_Only_Retrieval = MultiModalRAG("llama4")
    Text_Image_Retrieval = MultiModalRAG("")

