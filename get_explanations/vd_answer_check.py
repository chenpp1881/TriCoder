import re
import os
import pdb
import sys
import json
from tqdm import tqdm
import concurrent.futures
from prompt_templates import *
from vd_oai_model_interface import chat_gpt_text_completion, MODEL

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

task_id = "DefectPre"
datalines = {}

file_path = os.path.join(BASE_DIR, '../criteria', f'{task_id}', f'final_prompt_set_{MODEL}.json')
with open(file_path, 'r', encoding='utf-8') as f:
    FINAL_PROMPT_SET = json.load(f)

STEM_CORRECT_SYSTEM = create_explanation_generation_prompt(FINAL_PROMPT_SET)

STEM_CORRECT_USER = """**code:**
{code}
"""


def all_values_are_str(data: dict) -> bool:
    return all(isinstance(v, str) for v in data.values())


def check_answer(line):
    code = line['code']
    messages = [
        {'role': 'system', 'content': STEM_CORRECT_SYSTEM},
        {'role': 'user', 'content': STEM_CORRECT_USER.format(code=code)}
    ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            continue

        if answer == '':
            continue
        else:
            start_index = answer.find('{')
            end_index = answer.rfind('}')
            json_str = answer[start_index:end_index + 1]
            try:
                json_str = json.loads(json_str)
                assert all_values_are_str(json_str)
            except Exception as e:
                continue
            line.update(json_str)
            break
    return line


if __name__ == '__main__':
    with open(f"../Datasets/{task_id}/extracted_data.jsonl", 'r') as f:
        datalines = [json.loads(l) for l in f if l.strip()]

    with open(f'../Filtered_DS/{task_id}/{MODEL}.jsonl', 'w', encoding='utf-8') as f:
        with concurrent.futures.ThreadPoolExecutor(250) as executor:
            futures = [executor.submit(check_answer, item) for item in datalines]
            results = []
            for ind, future in enumerate(tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
                f.write(json.dumps(future.result(), ensure_ascii=False) + '\n')
