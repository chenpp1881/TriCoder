import os
import time
from utils import *
from tqdm import tqdm
from functools import partial
from typing import List, Dict, Set
from get_explanations.prompt_templates import *
from concurrent.futures import ThreadPoolExecutor
from get_explanations.vd_oai_model_interface import chat_gpt_text_completion, MODEL

LANGUAGES = {
    "DefectPre": "C",
    "Devign": "C",
    "Reveal": "C",
    "POJ-104": "C/C++",
    "Authorship": "Python",
}

task_definitions = {
    'POJ-104': """**Task Name:** Programming Problem Recognition

**Task Description:** This task aims to identify which algorithmic problem a given code snippet is attempting to solve. The task involves classifying source code submissions into one of 104 distinct programming problems from an online judge system. Each code sample represents a student's solution to a specific algorithmic challenge.

**Classification Type:** 104-class classification
- Each class represents a unique programming problem
- Classes are identified by problem IDs (0-103)
- Each problem typically involves different algorithmic concepts (e.g., sorting, graph algorithms, dynamic programming, etc.)

**Programming Language:** C/C++""",


    'Authorship': """**Task Name:** Authorship Attribution

**Task Description:** This task focuses on correctly identifying the author of a given code snippet by analyzing coding styles, patterns, and individual programming preferences. The challenge lies in detecting subtle variations in coding practices that distinguish different developers, even when solving the same problems.

**Classification Type:** 66-class classification
- Each class represents a unique author
- Classes are identified by author IDs (0-65)
- Each author has their distinct coding style, including naming conventions, indentation preferences, algorithm implementation choices, and code structure patterns

**Programming Language:** Python""",


    'DefectPre': """**Task Name:** Defect Prediction

**Task Description:** This task aims to predict whether a code snippet contains defects or will produce incorrect results when executed. The task helps identify problematic code that may fail during runtime or produce wrong outputs, based on submissions to competitive programming platforms.

**Classification Type:** 4-class classification
- OK: Code runs correctly and produces expected output
- Wrong Answer: Code executes but produces incorrect results
- Time Limit Exceeded: Code is too slow and exceeds the allowed execution time
- Runtime Error: Code crashes or encounters errors during execution

**Programming Language:** C""",
    
    
    'Devign': """**Task Name:** Vulnerability Detection

**Task Description:** This task involves identifying whether a code snippet contains security vulnerabilities that could potentially be exploited. The goal is to automatically detect code weaknesses that may lead to security breaches, helping developers identify and fix security issues early in the development process.

**Classification Type:** Binary classification
- Secure code: No vulnerabilities detected
- Vulnerable code: Code contains security vulnerabilities

**Programming Language:** C""",


    'Reveal': """**Task Name:** Vulnerability Detection

**Task Description:** This task involves identifying whether a code snippet contains security vulnerabilities that could potentially be exploited. The goal is to automatically detect code weaknesses that may lead to security breaches, helping developers identify and fix security issues early in the development process.

**Classification Type:** Binary classification
- Secure code: No vulnerabilities detected
- Vulnerable code: Code contains security vulnerabilities

**Programming Language:** C"""
}

GENERAL_CRITERIA = [
    {
        'name': 'Basic Functionality Interpretation',
        'rationale': 'Explain the primary purpose and functionality of the code. What problem does it solve at a high level?'
    },
    {
        'name': 'Logic and Flow Interpretation',
        'rationale': 'Describe the execution logic and control flow of the code. How does it achieve its purpose step-by-step, including key branches, loops, and decision points?'
    },
    {
        'name': 'External Dependency Analysis',
        'rationale': 'Detail all interactions the code has with the outside world. What external functions, libraries, or APIs does it call? What external state (like databases, files, or contract storage) does it read from or write to?'
    },
]


def get_task_definition(task_name):
    if task_name == "DefectPre":
        sampled_data = sample_k_examples_per_label("Datasets/DefectPre/extracted_data.jsonl", 3)
        samples_block = format_samples_for_prompt(sampled_data, LANGUAGES['DefectPre'])
    elif task_name == "Devign":
        task_name += " Vulnerability Detection"
        sampled_data = sample_k_examples_per_label("Datasets/Devign/extracted_data.jsonl", 3)
        samples_block = format_samples_for_prompt(sampled_data, LANGUAGES['Devign'])
    elif task_name == "Reveal":
        task_name += " Vulnerability Detection"
        sampled_data = sample_k_examples_per_label("Datasets/Reveal/extracted_data.jsonl", 3)
        samples_block = format_samples_for_prompt(sampled_data, LANGUAGES['Reveal'])
    else:
        task_name = "Smart Contract Vulnerability Detection"
        sampled_data = sample_k_examples_per_label("Datasets/SCVD/extracted_data.jsonl", 3)
        samples_block = format_samples_for_prompt(sampled_data, LANGUAGES['SCVD'])

    messages = [
        {'role': 'system', 'content': TASK_DESC_PROMPT_SYSTEM},
        {'role': 'user', 'content': TASK_DESC_PROMPT_USER(task_name, samples_block)}
    ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            continue

        if answer == '':
            continue
        else:
            extracted_dict = extract_task_description_dict(answer)
            if extracted_dict is None:
                continue
            else:
                return extracted_dict


def get_general_criteria(task_desc):
    messages = [
        {'role': 'system', 'content': GET_PROMPT_SYSTEM},
        {'role': 'user', 'content': GET_PROMPT_USER(task_desc)}
    ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            continue

        if answer == '':
            continue
        else:
            extracted_dict = extract_and_check_dict1(answer)
            if extracted_dict is None:
                continue
            else:
                return extracted_dict["specific_perspectives"]


def get_removal_criteria(general_criteria, specific_criteria):
    messages = [
        {'role': 'system', 'content': FILTER_PROMPT1_SYSTEM},
        {'role': 'user', 'content': FILTER_PROMPT1_USER(general_criteria, specific_criteria)}
    ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            print(f'Request again: {e}')
            continue

        if answer == '':
            print('Request again: empty output')
            continue
        else:
            extracted_dict = extract_and_check_dict2(answer)
            if extracted_dict is None:
                print('Request again: wrong format')
                continue
            else:
                print(extracted_dict)
                return extracted_dict["perspectives_to_remove"]


def get_merged_criteria(filtered_criteria):
    messages = [
        {'role': 'system', 'content': FILTER_PROMPT2_SYSTEM},
        {'role': 'user', 'content': FILTER_PROMPT2_USER(filtered_criteria)}
    ]
    while True:
        try:
            answer = chat_gpt_text_completion(messages=messages)
        except Exception as e:
            print(f'Request again: {e}')
            continue

        if answer == '':
            print('Request again: empty output')
            continue
        else:
            extracted_dict = extract_and_check_dict1(answer)
            if extracted_dict is None:
                print('Request again: wrong format')
                continue
            else:
                print(extracted_dict)
                return extracted_dict["specific_perspectives"]


def remove_criteria(all_perspectives, names_to_remove):
    if not names_to_remove:
        return all_perspectives[:]

    if not all_perspectives:
        return []

    removal_set: Set[str] = set(names_to_remove)
    filtered_list = [
        perspective
        for perspective in all_perspectives
        if perspective.get('name') not in removal_set
    ]

    return filtered_list


def _score_single_sample(sample, task_description, language, perspective_name, perspective_rationale):
    code = sample.get('code', '')
    if not code:
        return None

    messages = [
        {'role': 'system', 'content': SCORING_PROMPT_SYSTEM},
        {'role': 'user', 'content': SCORING_PROMPT_USER(
            task_description, language, code, perspective_name, perspective_rationale
        )}
    ]

    max_retries = 10
    for attempt in range(max_retries):
        try:
            answer = chat_gpt_text_completion(messages=messages)
            score = extract_and_check_score(answer)
            if score is not None:
                return score
        except Exception as e:
            logger.error(f"{e}")

        if attempt < max_retries - 1:
            time.sleep(1)
    return None


def extract_and_check_score(raw_text):
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None
    try:
        start_index = raw_text.find('{')
        end_index = raw_text.rfind('}')
        if start_index == -1 or end_index == -1:
            return None
        json_str = raw_text[start_index: end_index + 1]
        data = json.loads(json_str)
        if isinstance(data, dict) and 'relevance_score' in data and isinstance(data['relevance_score'], int):
            score = data['relevance_score']
            if 1 <= score <= 5:
                return score
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    return None


def scoring_and_select_criteria(task_description, candidate_criteria, dataset_path, sample_num, top_n, language,
                                max_workers):
    if not candidate_criteria:
        return []


    try:
        sampled_data = sample_jsonl(dataset_path, k=sample_num, random_seed=42)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"{e}")
        return []

    criteria_scores = []

    for criteria in tqdm(candidate_criteria, desc="Overall Scoring Progress"):
        perspective_name = criteria.get('name', 'N/A')
        perspective_rationale = criteria.get('rationale', '')

        all_scores = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            score_func_with_context = partial(
                _score_single_sample,
                task_description=task_description,
                language=language,
                perspective_name=perspective_name,
                perspective_rationale=perspective_rationale
            )

            results_iterator = executor.map(score_func_with_context, sampled_data)
            all_scores = list(
                tqdm(results_iterator, total=len(sampled_data), desc=f"Scoring '{perspective_name[:20]}...'"))

        valid_scores = [s for s in all_scores if s is not None]

        if valid_scores:
            average_score = sum(valid_scores) / len(valid_scores)
            criteria_scores.append({'criteria': criteria, 'avg_score': average_score})

    sorted_scores = sorted(criteria_scores, key=lambda x: x['avg_score'], reverse=True)

    top_criteria_with_scores = sorted_scores[:top_n]
    top_criteria = [item['criteria'] for item in top_criteria_with_scores]

    return top_criteria


if __name__ == '__main__':
    task_id = "DefectPre"
    task_desc = task_definitions[task_id]
    logger.info(f"task_id = {task_id}")
    logger.info(task_desc)

    SPECIFIC_CRITERIA = get_general_criteria(task_desc)
    logger.info(SPECIFIC_CRITERIA)

    names_to_remove = get_removal_criteria(GENERAL_CRITERIA, SPECIFIC_CRITERIA)
    logger.info(names_to_remove)
    FILTERED_SPECIFIC_CRITERIA = remove_criteria(SPECIFIC_CRITERIA, names_to_remove)
    logger.info(FILTERED_SPECIFIC_CRITERIA)

    MERGED_SPECIFIC_CRITERIA = get_merged_criteria(FILTERED_SPECIFIC_CRITERIA)
    logger.info(MERGED_SPECIFIC_CRITERIA)

    FINAL_TASK_CRITERIA = scoring_and_select_criteria(
        task_description=task_desc,
        candidate_criteria=MERGED_SPECIFIC_CRITERIA,
        dataset_path=f"Datasets/{task_id}/extracted_data.jsonl",
        sample_num=50,
        top_n=3,
        language=LANGUAGES[task_id],
        max_workers=50
    )

    FINAL_PROMPT_SET = GENERAL_CRITERIA + FINAL_TASK_CRITERIA

    with open(f'./perspectives/{task_id}/final_prompt_set_{MODEL}.json', 'w', encoding='utf-8') as f:
        json.dump(FINAL_PROMPT_SET, f, indent=4)
