from data import data_loading, create_nodes, image_to_base64
from prompts import retrieval_prompt, text_prompt, image_prompt, evaluation_prompt
from retriever import multimodal_vector_db, embeddings
import tempfile
from llama_index.core.schema import TextNode, ImageNode
import base64
from llama_index.core.llms import ChatMessage, TextBlock, ImageBlock
from llama_index.llms.ollama import Ollama
import json, os, re

class MultiModalRAG:
    retrieval_type: str; benchmark: str
    size: int

    def __init__(self, mllm, retrieval_type, benchmark, size):
        self.mllm = Ollama(
            model=mllm,
            request_timeout=600.0
        )
        self.retrieval_type = retrieval_type
        self.benchmark = benchmark
        self.size = size
        self.total_samples = 0
        self.total_score = 0
        self.folder_name = self.create_output_folder()
        self.pipeline(self.benchmark)

    def create_output_folder(self):
        base_folder = "output"
        counter = 1

        while True:
            if counter == 1:
                folder_name = base_folder
            else:
                folder_name = f"{base_folder}_{counter}"

            if not os.path.exists(folder_name):
                os.makedirs(folder_name)
                print(f"Created output folder: {folder_name}")
                return folder_name
            counter += 1


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
                    return res_node.node.image
                else:
                    print("ImageNode found but no image data available")
                    return None

    def evaluation(self, response):
        eval_prompt = evaluation_prompt.format(response_text=response)
        eval_mllm = Ollama( model="gemma3", request_timeout=600.0)
        messages = [
            ChatMessage(role="user", blocks=[TextBlock(text=eval_prompt)])
        ]
        eval_result = eval_mllm.chat(messages)

        eval_result = str(eval_result).strip()

        if '```' in eval_result:
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', eval_result, re.DOTALL)
            if json_match:
                eval_result = json_match.group(1)
            else:
                eval_result = re.sub(r'```[a-z]*\s*', '', eval_result)
                eval_result = re.sub(r'```', '', eval_result)

        eval_result = json.loads(eval_result)

        print(f"Evaluation: {eval_result}")

        return eval_result

    def save(self, messages, response, eval_result):

        score = eval_result.get('score')

        self.total_samples += 1
        self.total_score += score

        save_data = {
            "sample_number": self.total_samples,
            "model": str(self.mllm.model),
            "messages": str(messages),
            "response": str(response),
            "score": score,
            "eval_result": eval_result,
        }
        output_file = os.path.join(self.folder_name, f"{self.retrieval_type}_retrieval.jsonl")

        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

        if self.total_samples % self.size == 0:
            accuracy = (self.total_score / self.total_samples) * 100
            bias_score = 100 - accuracy
            print(f"Accuracy: {accuracy}")
            print(f"Bias Score: {bias_score}")

            metrics = {
                "accuracy": accuracy,
                "bias_score": bias_score
            }

            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + '\n')

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
                benchmark_metadata=benchmark_metadata,
                ans0=ans0,
                ans1=ans1,
                ans2=ans2,
            )

            messages = []

            match self.retrieval_type:
                case "text_to_text":
                    context = self.text_retrieval(retrieval_text_prompt)

                    Final_MLLM_prompt = text_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        benchmark_metadata=benchmark_metadata,
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
                case "image_to_image":
                    context = self.image_retrieval(benchmark_image)
                    print(f"Image data type: {type(context)}")
                    print(f"Image data length: {len(context) if context else 0}")

                    Final_MLLM_prompt = image_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        benchmark_metadata=benchmark_metadata,
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
                case "text_to_image":
                    pass
                case "text_to_both":
                    text_context = self.text_retrieval(retrieval_text_prompt)
                    image_context = self.image_retrieval(benchmark_image)

                    Final_MLLM_prompt = text_prompt.format(
                        benchmark_context=benchmark_context,
                        benchmark_question=benchmark_question,
                        benchmark_metadata=benchmark_metadata,
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
                print("Response: ", response)
                eval_result = self.evaluation(response)
                self.save(messages, response, eval_result)

if __name__ == "__main__":
    corpus = data_loading("HuggingFaceM4/OBELICS", "train", True, 10, False)
    nodes = create_nodes(corpus)
    storage_context = multimodal_vector_db()
    index = embeddings(storage_context, nodes)

    benchmark = data_loading("ucf-crcv/SB-Bench", "real", True, 5, False)

    Text_To_Text_Retrieval = MultiModalRAG("llava", "text_to_text", benchmark, 5)
    Image_To_Image_Retrieval = MultiModalRAG("llava", "image_to_image", benchmark, 5)
    Text_To_Both_Retrieval = MultiModalRAG("llava", "text_to_both", benchmark, 5)
