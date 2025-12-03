from llama_index.core import PromptTemplate

retrieval_text_prompt = (
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Option 1: {ans0}\n"
    "Option 2: {ans1}\n"
    "Option 3: {ans2}\n"
    "Answer: "
)

text_prompt = PromptTemplate(retrieval_text_prompt)
