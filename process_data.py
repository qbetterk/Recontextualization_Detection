#!/usr/bin/env python3
#
import sys, os, pdb
import json
import random

random.seed(42)

def _load_json(path=None):
    if path is None or not os.path.exists(path):
        raise IOError('File does not exist: %s' % path)
        # return None
    with open(path) as df:
        data = json.loads(df.read())
    return data


def _save_json(data, path=None):
    print(f"saving to {path} ......")
    with open(path, "w") as tf:
        json.dump(data, tf, indent=2)


def main():
    source_data_path = "./data_my/newsroom_sample.json"
    data = _load_json(source_data_path)
    random.shuffle(data)
    train_data, valid_data = data[:int(len(data)*0.8)], data[int(len(data)*0.8)+1:]
    _save_json(train_data, "./data_my/newsroom_train.json")
    _save_json(valid_data, "./data_my/newsroom_val.json")

if __name__ == "__main__":
    main()