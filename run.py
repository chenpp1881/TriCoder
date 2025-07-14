import random
import numpy as np
import torch
import argparse
from data_utils import load_data, MODEL
from train import Trainer
import os
import sys
import logging
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), "logs")
os.makedirs(logs_dir, exist_ok=True)
cur_time = datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = datetime.now().strftime(f"run_{cur_time}.log")
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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    return


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr_IF', type=float, default=1e-5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--model_path', type=str, default='codet5-base')
    parser.add_argument('--max_length', type=int, default=1024)
    parser.add_argument('--savepath', type=str, default=f'./Results/{cur_time}')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--resume_file', type=str, default=None)
    parser.add_argument('--exp_cfg', type=str, default='6')
    parser.add_argument('--task_id', type=str, default='Devign')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)

    args = parse_args()
    logger.info(vars(args))

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device is %s, Model is %s', args.device, MODEL)

    set_seed(args.seed)

    if args.task_id not in ['SCVD', 'Devign', 'DefectPre', 'Reveal', 'POJ-104', 'Authorship']:
        logger.error(f"task_id is {args.task_id}")
        sys.exit(1)

    train_set, test_set = load_data(args)

    trainer = Trainer(args)
    trainer.train_classicication([train_set, test_set])
