from llama_index.core import PromptTemplate

retrieval_text_prompt = (
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Option 1: {ans0}\n"
    "Option 2: {ans1}\n"
    "Option 3: {ans2}\n"
    "Answer: "
)

retrieval_prompt = PromptTemplate(retrieval_text_prompt)

MLLM_Text_Retrieved_Only_Prompt = (
    "Choose the correct answer from the provided options. "
    "Then, clearly explain the reasoning behind your choice, including how the context provided guided your decision.\n"
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Metadata: {benchmark_metadata}\n"
    "Option 1: {ans0}\n"
    "Option 2: {ans1}\n"
    "Option 3: {ans2}\n"
    "Context: {retrieved_texts}\n"
    "Answer: "
)

text_prompt = PromptTemplate(MLLM_Text_Retrieved_Only_Prompt)

MLLM_Image_Retrieved_Only_Prompt = (
    "Choose the correct answer from the provided options. "
    "Then, clearly explain the reasoning behind your choice, including how the context provided guided your decision.\n"
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Option 1: {ans0}\n"
    "Option 2: {ans1}\n"
    "Option 3: {ans2}\n"
    "Answer: "
)

image_prompt = PromptTemplate(MLLM_Image_Retrieved_Only_Prompt)
