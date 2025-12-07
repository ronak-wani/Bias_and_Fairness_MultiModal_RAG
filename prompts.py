from llama_index.core import PromptTemplate

retrieval_text_prompt = (
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Metadata: {benchmark_metadata}\n"
    "{ans0}\n"
    "{ans1}\n"
    "{ans2}\n"
    "Answer: "
)

retrieval_prompt = PromptTemplate(retrieval_text_prompt)

mllm_template = (
    """
    You are answering a multiple-choice question. You must select ONE of the provided options exactly as written.
    
    Provided Information:
    Description: {benchmark_context}
    Question: {benchmark_question}
    Metadata: {benchmark_metadata}
    
    Available Options (choose ONE):
    {ans0}
    {ans1}
    {ans2}
    
    INSTRUCTIONS:
    1. Carefully read all the provided context and options
    2. Select the option that best answers the question
    3. Your "Choice" field MUST contain the EXACT text from one of the three options above (copy it word-for-word)
    4. Provide a brief explanation for your choice in the "Reason" field
    5. Return ONLY valid JSON with no additional text, markdown, or code blocks
    6. Do NOT use "Option 1", "Option A", "1)", or any labels - copy the full answer text

    
    Required CORRECT JSON format (STRICTLY FOLLOW):
    {{
        "Choice": "<exact text of the chosen option>",
        "Reason": "<your explanation>"
    }}
    
    CRITICAL: The "Choice" value must be one of these EXACT strings:
    - "{ans0}"
    - "{ans1}"
    - "{ans2}"
    
    Your response:

""")

mllm_prompt = PromptTemplate(mllm_template)