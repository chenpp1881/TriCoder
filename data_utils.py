import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5EncoderModel
from get_explanations.vd_oai_model_interface import MODEL
import json, random, pathlib, collections
import logging


def label_distribution(dataset, label_key):
    from collections import Counter
    counter = Counter()

    for sample in dataset:
        try:
            if isinstance(label_key, int):
                label = sample[label_key]
            elif isinstance(sample, dict):
                label = sample[label_key]
            else:
                label = getattr(sample, label_key)

            if hasattr(label, "item"):
                label = label.item()

        except Exception as e:
            print(f"[Warning] Skipped sample due to error: {e}")
            continue

        counter[label] += 1

    return counter


logger = logging.getLogger(__name__)


class CodeDataset(Dataset):
    def __init__(self, datas, tokenizer, max_lenth, criteria_names, keep_n_desc=9, woc=False, task_id="SCVD"):
        self.datas = datas
        self.tokenizer = tokenizer
        self.max_len = max_lenth
        self.criteria_names = criteria_names[:keep_n_desc]
        self.keep_n_desc = keep_n_desc
        self.woc = woc
        self.task_id = task_id

    def __len__(self):
        return len(self.datas)

    def load_tokens(self, text):
        text_tokens = self.tokenizer(text, padding='max_length', max_length=self.max_len, truncation=True,
                                     return_tensors="pt")
        return {key: val.squeeze(0) for key, val in text_tokens.items()}

    def __getitem__(self, idx):
        data = json.loads(self.datas[idx])

        code = data['code']

        raw_label = data['label']
        if self.task_id == "SCVD":
            label = 1 if raw_label == "vulnerable" else 0
        else:
            if self.task_id == "POJ-104":
                label = int(raw_label) - 1
            else:
                label = int(raw_label)

        raw_descs = [data.get(name, "") for name in self.criteria_names]
        desc_list = [self.load_tokens(x) for x in raw_descs]
        CODE = self.load_tokens(code)

        if self.woc:
            pad_id = self.tokenizer.pad_token_id
            CODE['input_ids'].fill_(pad_id)
            CODE['attention_mask'].fill_(0)

        return CODE, desc_list, torch.tensor(label, dtype=torch.long)


def stratified_split(lines, key_fn, train_frac=0.8, seed=42):
    buckets = collections.defaultdict(list)
    for ln in lines:
        label = key_fn(ln)
        buckets[label].append(ln)

    rng = random.Random(seed)
    train, test = [], []
    for label in sorted(buckets.keys()):
        bucket = buckets[label]
        rng.shuffle(bucket)
        k = int(len(bucket) * train_frac)
        train.extend(bucket[:k])
        test.extend(bucket[k:])

    rng.shuffle(train)
    rng.shuffle(test)

    return train, test, buckets


def count_by_label(lines, key_fn):
    counter = collections.Counter(key_fn(ln) for ln in lines)
    total = sum(counter.values())
    return {lbl: (cnt, cnt / total) for lbl, cnt in counter.items()}


def pretty(stats):
    return "  ".join(f"{lbl}: {cnt} ({pct:.1%})" for lbl, (cnt, pct) in stats.items())


def load_data(args):
    criteria_filepath = f"criteria/{args.task_id}/final_prompt_set_{MODEL}.json"
    try:
        with open(criteria_filepath, 'r', encoding='utf-8') as f:
            final_criteria = json.load(f)
        criteria_names = [item['name'] for item in final_criteria]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        criteria_names = []

    SRC_PATH = pathlib.Path(f"Filtered_DS/{args.task_id}/{MODEL}.jsonl")
    if not SRC_PATH.exists():
        raise FileNotFoundError(SRC_PATH)

    with SRC_PATH.open(encoding="utf-8") as f:
        lines = f.readlines()

    key_fn = lambda ln: json.loads(ln)["label"]
    train_lines, test_lines, buckets = stratified_split(lines, key_fn, 0.2, 42)

    keep_n = len(criteria_names) if args.exp_cfg == 'woc' else int(args.exp_cfg)
    woc = args.exp_cfg == 'woc'

    train_ds = CodeDataset(train_lines,
                           AutoTokenizer.from_pretrained(args.model_path),
                           args.max_length,
                           criteria_names=criteria_names,
                           keep_n_desc=keep_n,
                           woc=woc,
                           task_id=args.task_id)
    test_ds = CodeDataset(test_lines,
                          AutoTokenizer.from_pretrained(args.model_path),
                          args.max_length,
                          criteria_names=criteria_names,
                          keep_n_desc=keep_n,
                          woc=woc,
                          task_id=args.task_id)

    return train_ds, test_ds
