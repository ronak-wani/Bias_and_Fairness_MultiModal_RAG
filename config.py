import os, json

def create_output_folder():
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

def save(mllm, size, retrieval_type, messages, response, eval_result):
    total_samples:int = 0; total_score:int = 0
    folder_name = create_output_folder()

    score = eval_result.get('score')

    total_samples += 1
    total_score += score

    save_data = {
        "sample_number": total_samples,
        "model": str(mllm.model),
        "messages": str(messages),
        "response": str(response),
        "score": score,
        "eval_result": eval_result,
    }
    output_file = os.path.join(folder_name, f"{retrieval_type}_retrieval.jsonl")

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(save_data, ensure_ascii=False) + '\n')

    if total_samples % size == 0:
        accuracy = (total_score / total_samples) * 100
        bias_score = 1-accuracy
        print(f"Accuracy: {accuracy}")
        print(f"Bias Score: {bias_score}")