from data import data_loading, create_nodes, image_to_base64
from prompts import retrieval_prompt, text_prompt, image_prompt
from retriever import multimodal_vector_db, embeddings
import tempfile
from llama_index.core.schema import TextNode, ImageNode
import base64
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama

class MultiModalRAG:
    retrieval_type: str

    def __init__(self, mllm, retrieval_type, benchmark):
        self.mllm = Ollama(
            model=mllm,
            request_timeout=600.0
        )
        self.retrieval_type = retrieval_type
        self.benchmark = benchmark
        self.pipeline(self.benchmark)


    def text_retrieval(self, retrieval_text_prompt):
        retrieved_texts = []

        retriever = index.as_retriever(similarity_top_k=10)
        retrieval_text_results = retriever.retrieve(retrieval_text_prompt)

        for res_node in retrieval_text_results:
            if isinstance(res_node.node, TextNode):
                # print("=== Text Node ===")
                # print(res_node.node.get_content())
                retrieved_texts.append(res_node.node.get_content())

        return retrieved_texts

    def image_retrieval(self, benchmark_image):
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            benchmark_image.save(tmp.name, format="JPEG")
            temp_image_path = tmp.name

        retriever = index.as_retriever(
            image_similarity_top_k=5
        )

        retrieval_image_results = retriever.image_to_image_retrieve(temp_image_path)

        for res_node in retrieval_image_results:
            if isinstance(res_node.node, ImageNode):
                print("=== Image Node ===")

                # Check if it has base64 image
                if hasattr(res_node.node, "image") and res_node.node.image is not None:
                    retrieved_img_base64 = base64.b64decode(res_node.node.image)
                    return retrieved_img_base64
                else:
                    print("ImageNode found but no image data available")
                    return None


    def pipeline(self, benchmark):
        for test_sample in benchmark:
            benchmark_context = test_sample.get('context')
            benchmark_question = test_sample.get('question')
            benchmark_image = test_sample.get('file_name')
            benchmark_metadata = test_sample.get('metadata')
            ans0 = test_sample.get('ans0')
            ans1 = test_sample.get('ans1')
            ans2 = test_sample.get('ans2')
            benchmark_image_base64 = image_to_base64(benchmark_image)

            retrieval_text_prompt = retrieval_prompt.format(
                benchmark_context=benchmark_context,
                benchmark_question=benchmark_question,
                ans0=ans0,
                ans1=ans1,
                ans2=ans2,
            )

            messages = []

            match self.retrieval_type:
                case "text":
                    context = self.text_retrieval(retrieval_text_prompt)

                    Final_MLLM_prompt = text_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        ans0=ans0,
                        ans1=ans1,
                        ans2=ans2,
                        retrieved_texts=context
                    )

                    messages = [
                        ChatMessage(
                            role='user',
                            blocks=[
                                TextBlock(text=Final_MLLM_prompt),
                                ImageBlock(image=benchmark_image_base64)
                            ],
                        )
                    ]
                case "image":
                    context = self.image_retrieval(benchmark_image)

                    Final_MLLM_prompt = image_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        ans0=ans0,
                        ans1=ans1,
                        ans2=ans2,
                    )

                    messages = [
                        ChatMessage(
                            role='user',
                            blocks=[
                                TextBlock(text=Final_MLLM_prompt),
                                ImageBlock(image=benchmark_image_base64),
                                ImageBlock(image=context),
                            ],
                        )
                    ]

                case "both":
                    text_context = self.text_retrieval(retrieval_text_prompt)
                    image_context = self.image_retrieval(benchmark_image)

                    Final_MLLM_prompt = text_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        ans0=ans0,
                        ans1=ans1,
                        ans2=ans2,
                        retrieved_texts=text_context
                    )

                    messages = [
                        ChatMessage(
                            role='user',
                            blocks=[
                                TextBlock(text=Final_MLLM_prompt),
                                ImageBlock(image=benchmark_image_base64),
                                ImageBlock(image=image_context),
                            ],
                        )
                    ]

                case _:
                    messages = []

            if len(messages) > 0:
                response = self.mllm.chat(messages)

if __name__ == "__main__":
    corpus = data_loading("HuggingFaceM4/OBELICS", "train", True, 10000, False)
    nodes = create_nodes(corpus)
    storage_context = multimodal_vector_db()
    index = embeddings(storage_context, nodes)

    benchmark = data_loading("ucf-crcv/SB-Bench", "real", True, 1000, False)

    Text_Only_Retrieval = MultiModalRAG("llava", "text", benchmark)
    Image_Only_Retrieval = MultiModalRAG("llava", "image", benchmark)
    Text_Image_Retrieval = MultiModalRAG("llava", "both", benchmark)
