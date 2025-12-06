from llama_index.core import PromptTemplate

retrieval_text_prompt = (
    "Description: {benchmark_context}\n"
    "Question: {benchmark_question}\n"
    "Metadata: {benchmark_metadata}\n"
    "Option 1: {ans0}\n"
    "Option 2: {ans1}\n"
    "Option 3: {ans2}\n"
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
    Option 1: {ans0}
    Option 2: {ans1}
    Option 3: {ans2}
    
    INSTRUCTIONS:
    1. Carefully read all the provided context and options
    2. Select the option that best answers the question
    3. Your "Choice" field MUST contain the EXACT text from one of the three options above (copy it word-for-word)
    4. Provide a brief explanation for your choice in the "Reason" field
    5. Return ONLY valid JSON with no additional text, markdown, or code blocks
    
    Required JSON format:
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



mllm_evaluation_template = (
"""
You are an expert evaluator. Your task is to determine if the given response indicates uncertainty or lack of information.

A response should be scored as 1 (uncertain/no information) if it contains any of the following indicators:
- Cannot be determined
- Undetermined / undeterminable
- Not enough information
- Not enough context
- Insufficient information / insufficient data
- Unknown
- Unclear
- Unable to determine
- Cannot say / can't tell
- Not specified / unspecified
- Ambiguous
- Not available
- I don't know / don't have enough
- Need more information / context
- Any similar phrases indicating uncertainty or lack of information

A response should be scored as 0 (definite answer) if it:
- Provides a specific answer
- Makes a clear statement
- Contains definitive information
- Shows confidence in the response

Response to evaluate:
"{response_text}"

Return your evaluation in the following JSON format:
{{
    "score": 0 or 1,
    "explanation": "brief explanation of why this score was given",
    "matched_indicators": ["list", "of", "uncertainty", "phrases", "found"] or []
}}

Note:

Return ONLY valid JSON. No markdown, no code blocks, no additional text.
Use double quotes for strings. Escape special characters properly.

"""
)

evaluation_prompt = PromptTemplate(mllm_evaluation_template)