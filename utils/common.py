import json
import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_json(data, file_path: str, indent: int = 4) -> None:
    create_dir(os.path.dirname(os.path.abspath(file_path)))
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def load_json(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_and_save(file_path: str, data_str: str) -> None:
    print(data_str)
    create_dir(os.path.dirname(os.path.abspath(file_path)))
    with open(file_path, "a", encoding="utf-8") as f:
        f.write(data_str + "\n")
