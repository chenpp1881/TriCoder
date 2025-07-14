import json
import random
import os
import re
import json
import logging
from datetime import datetime
from typing import List, Set
import json

from typing import Dict, Any
import random
import collections

logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)
cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = datetime.now().strftime(f"generate_prompt_{cur_time}.log")
log_filepath = os.path.join(logs_dir, log_filename)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

file_handler = logging.FileHandler(log_filepath, encoding="utf-8")
file_fmt = logging.Formatter("%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_fmt)
logger.addHandler(file_handler)

console_handler = logging.StreamHandler()
console_fmt = logging.Formatter("%(asctime)-15s %(levelname)s: %(message)s")
console_handler.setFormatter(console_fmt)
logger.addHandler(console_handler)

logger = logging.getLogger(__name__)


def extract_and_check_dict1(raw_text: str) -> dict | None:
    match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_text, re.DOTALL)
    if not match:
        return None

    json_str = match.group(1).strip()

    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None

    if not isinstance(parsed_data, dict):
        return None

    if 'specific_perspectives' not in parsed_data:
        return None

    if not isinstance(parsed_data['specific_perspectives'], list):
        return None

    return parsed_data


def extract_and_check_dict2(raw_text: str) -> dict | None:
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    start_index = raw_text.find('{')
    end_index = raw_text.rfind('}')

    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None

    json_str = raw_text[start_index: end_index + 1]

    try:
        parsed_data = json.loads(json_str)
    except json.JSONDecodeError as e:
        return None

    if not isinstance(parsed_data, dict):
        return None

    key_to_check = 'perspectives_to_remove'
    if key_to_check not in parsed_data:
        return None

    removal_list = parsed_data[key_to_check]
    if not isinstance(removal_list, list):
        return None

    if not all(isinstance(item, str) for item in removal_list):
        return None

    return parsed_data


def sample_jsonl(input_filepath, k=1, random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)

    with open(input_filepath, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]

    if k <= 0:
        return []

    selected_lines = random.sample(lines, k)

    return [json.loads(line) for line in selected_lines]


def get_all_labels_from_jsonl(filepath: str) -> Set[str]:
    labels = set()
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                if 'label' in data:
                    labels.add(str(data['label']))
            except (json.JSONDecodeError, KeyError):
                continue
    return labels


def sample_k_examples_per_label(filepath: str, k: int, random_seed: int = 42) -> Dict[str, List[Dict[str, Any]]]:
    random.seed(random_seed)

    buckets = collections.defaultdict(list)
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                label = str(data.get('label'))
                buckets[label].append(data)
            except (json.JSONDecodeError, KeyError):
                continue

    sampled_data = {}
    for label, items in buckets.items():
        if len(items) < k:
            sampled_data[label] = items
        else:
            sampled_data[label] = random.sample(items, k)

    return sampled_data


def format_samples_for_prompt(sampled_data: Dict[str, List[Dict[str, Any]]], language: str) -> str:
    prompt_str = ""
    sorted_labels = sorted(sampled_data.keys())

    for label in sorted_labels:
        prompt_str += f"**Label: `{label}` Samples:**\n"
        samples = sampled_data[label]
        for i, sample in enumerate(samples):
            code = sample.get('code', '# No code provided')
            prompt_str += f"```{language}\n// --- Sample {i + 1} ({label}) ---\n{code}\n```\n"
        prompt_str += "\n"

    return prompt_str.strip()


def extract_task_description_dict(raw_text: str) -> Dict[str, str] | None:
    match = re.search(r"```json\s*([\s\S]+?)\s*```", raw_text, re.DOTALL)
    if not match:
        return None

    try:
        data = json.loads(match.group(1))
        if isinstance(data, dict) and 'task_description' in data and isinstance(data['task_description'], dict):
            description_dict = data['task_description']
            required_keys = {"core_task", "context", "mechanism_and_impact", "non_goals"}
            if required_keys.issubset(description_dict.keys()):
                return description_dict
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    return None


def format_description_dict_to_prompt_str(description_dict: Dict[str, str]) -> str:
    if not description_dict:
        return "No task description provided."

    return (
        f"**A. The Core Task:**\n{description_dict.get('core_task', 'N/A')}\n\n"
        f"**B. The Context:**\n{description_dict.get('context', 'N/A')}\n\n"
        f"**C. The Problem's Mechanism & Impact:**\n{description_dict.get('mechanism_and_impact', 'N/A')}\n\n"
        f"**D. The Non-Goals:**\n{description_dict.get('non_goals', 'N/A')}"
    )