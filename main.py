#!/usr/bin/env python3
#
import sys, os, json
import pdb
from tqdm import tqdm

import numpy as np
import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

from trainer import SemaTrainer

np.random.seed(0)
torch.manual_seed(0)

class InConsistencyDetection(object):
    """docstring for InConsistencyDetection"""
    def __init__(self, arg=None):
        super(InConsistencyDetection, self).__init__()
        self.arg = arg
        self.data_dir = "./data/"
        self.cache_dir = "/local-scratch1/data/qkun/semafor/.cache/"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # params for debugging
        self.count_large, self.count_max = 0, 0
        

    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
            # return None
        with open(path) as df:
            data = json.loads(df.read())
        return data


    def _load_dir_json(self, dir_path=None):
        if dir_path is None or not os.path.exists(dir_path):
            raise IOError('Folder does not exist: %s' % dir_path)
        total_data = [] # assume data is a list of dialogs
        for filename in os.listdir(dir_path):
            if not filename.endswith(".json"): continue
            file_path = os.path.join(dir_path, filename)
            data = self._load_json(path=file_path)
            if type(data) == list:
                total_data.extend(data)
            else:
                total_data.append(data)
        return total_data


    def preprocess_function(self, examples):
        inputs = " ".join(examples["content"])
        model_inputs = self.tokenizer(inputs, return_tensors="pt", truncation=True, max_length=1024)
        labels = self.tokenizer(text_target=examples["title"], return_tensors="pt", truncation=True, max_length=128)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs


    def load_and_process_data(self, mode="train"):
        cache_path = os.path.join(self.cache_dir, mode)
        if os.path.exists(cache_path):
            tokenized_data = load_from_disk(cache_path)
        else:
            dataset = load_dataset(os.path.join(self.data_dir, mode), split="train")
            tokenized_data = dataset.map(self.preprocess_function)
            tokenized_data.save_to_disk(cache_path)
        return tokenized_data


    def load_model(self):
        # model_md = "t5-large"
        model_md = "google/flan-t5-large"
        self.tokenizer = AutoTokenizer.from_pretrained(model_md)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_md, device_map='auto')


    def train(self):
        train_data = self.load_and_process_data("train")
        val_data   = self.load_and_process_data("val")

        training_args = TrainingArguments(
            output_dir="/local-scratch1/data/qkun/semafor/results/",
            logging_dir="/local-scratch1/data/qkun/semafor/log/",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            warmup_steps=1e3,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=3,
            num_train_epochs=10,
            seed=0,
            fp16=False,
            auto_find_batch_size=True,
        )
        data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
        trainer = SemaTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            data_collator=data_collator
        )
        trainer.train()


def main():
    detect = InConsistencyDetection()
    detect.load_model()
    detect.train()

if __name__ == '__main__':
    main()