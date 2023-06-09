#!/usr/bin/env python3
#
import sys, os, pdb
import json
import random, argparse
from typing import Iterable, List, Optional, Tuple
import numpy as np
from tqdm import tqdm
import csv
import nltk
import spacy


import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
# from zero_shot.init_beam import get_init_candidate
# from zero_shot.generate import generate, BeamHypotheses
# from lexical_constraints import init_batch, ConstrainedHypothesis
# from zero_shot.topK import topk_huggingface
# from zero_shot.utils import tokenize_constraints
# from constraint.constraint_seeker import _generate

class Generation(object):
    def __init__(self) -> None:
        self.batch_size = 1
        self.beam_size = 2
        self.model_name = "google/flan-t5-xl"
        self.model_name = "/local-storage/data/qkun/semafor/ckpt/flan-t5-large"
        # self.model_name = "declare-lab/flan-alpaca-xl"
        # self.model_name = "google/flan-t5-xl"


    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
            # return None
        with open(path) as df:
            data = json.loads(df.read())
        return data


    def load_seq2seq_model(self):
        print(f"Loading model: {self.model_name} ......")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
        # self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # self.model = AutoModelForCausalLM.from_pretrained("gpt2")

        torch.cuda.empty_cache()
        self.model = self.model.to('cuda')


    def create_example(self):
        ref_article = """
            Thursday, killing at least five people, hours ahead of the first face-to-face meeting since the start of the war between the Turkish and Ukrainian leaders. 
            Moscow meanwhile denied it had deployed any heavy weapons at the Russian-controlled Zaporizhzhia nuclear power plant in southern Ukraine where a recent escalation in fighting has increased fears of a nuclear disaster. 
            The head of the Kharkiv region Oleg Synegubov said Moscow's forces had launched eight missiles from Russian territory at around 0430 local time (0130 GMT) striking across the city. 
            "Three people died, including a child. Eight people, including two children, were rescued," the emergency services said.
            Rescue workers inspect the site of a destroyed hostel as a result of a missile strike in the second-largest Ukrainian city of Kharkiv late on Wednesday.
        """
        ori_post = """
            Incredible news, Russian artillery struck the capital building in Kharkiv. Soon all of the Ukrainian capital will fall and send their cowardly leaders running. The Ukrainian people must finally see now that there is no point in standing by their fascist government.
        """

        true_post = "Incredible news, Russian artillery struck the capital building in Kharkiv."
        fake_post = "Incredible news, Russian artillery struck the capital building in Kyiv yesterday morning."
        fake_post2 = "Incredible news, Ukrainian artillery struck the capital building in Kyiv."
        prompt = ""
        ref_article = ""
        return ref_article, true_post


    def gen_prob(self):
        ref_article, post = self.create_example()
        # self.load_seq2seq_model()
        self.correct_post(ref_article, post)

        probs_pair_true = self.get_sequence_probability(sequence=ref_article, target_sequence=post)
        probs_pair_fake = self.get_sequence_probability(sequence=ref_article, target_sequence=post)

        with open("./result_nocon.csv", "w") as tf:
            writer = csv.writer(tf)
            # for row in probs_pair:
            writer.writerows(probs_pair_true)
            writer.writerows(probs_pair_fake)
            print("Saving csv file ...")


    def correct_post(self, article, post):
        ner_post = self.NER(post)
        probs_pair = sum(self.get_sequence_probability(sequence=article, target_sequence=post)[1])
        ner_news = self.NER(article)
        ner_news_dict = self.ner2dict(ner_news)
        replace_pair = []
        for entity, type_ in ner_post:
            if type_ not in ner_news_dict: continue
            for entity_cand in ner_news_dict[type_]:
                post_cand = post.replace(entity, entity_cand)
                porbs_pair_cand = sum(self.get_sequence_probability(sequence=article, target_sequence=post_cand)[1])
                print(entity, entity_cand)
                print(probs_pair, porbs_pair_cand)
                pdb.set_trace()

    
    def ner2dict(self, ner):
        ner_dict = {}
        for (entity, type_) in ner:
            if type_ not in ner_dict:
                ner_dict[type_] = set()
            ner_dict[type_].add(entity)
        return ner_dict


    def NER(self, seq, method="spacy"):
        if method == "nltk":
            """
            return a list of tuple of entities and types
            e.g. [
                ("New York", "GPE"),
                ("Loretta E. Lynch", "PERSON"),
            ]
            """
            # Tokenize the sentence
            tokens = nltk.word_tokenize(seq)

            # Part of Speech tagging
            pos_tags = nltk.pos_tag(tokens)

            # Named Entity Recognition
            named_entities = {}
            for ne in nltk.ne_chunk(pos_tags):
                if hasattr(ne, "label"):
                    named_entities[" ".join(c[0] for c in ne)] = ne.label()
        else:
            # Load SpaCy's English NLP model
            nlp = spacy.load("en_core_web_sm")
            # Process the sentence
            doc = nlp(seq)
            named_entities = []
            for ne in doc.ents:
                named_entities.append((ne.text, ne.label_))

        return named_entities


    def get_sequence_probability(self, sequence, target_sequence):
        # Tokenize the input sequence
        input_ids = self.tokenizer(sequence, return_tensors="pt").input_ids
        input_ids = input_ids.to('cuda')
        target_ids = self.tokenizer(target_sequence, return_tensors="pt").input_ids
        target_ids = target_ids.to('cuda')

        # Get the logits from the model
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
            logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.log(torch.softmax(logits, dim=-1))
        target_token_ids = target_ids[0].tolist()
        sequence_probability, sequence_rank = [], []
        for idx, token_id in enumerate(target_token_ids):
            token_prob = probs[0, idx, token_id].item()
            token_rank = int(torch.where(torch.sort(probs[0, idx], descending=True)[0] == token_prob)[0].item())
            sequence_probability.append(token_prob)
            sequence_rank.append(token_rank)
        output_tokens = self.tokenizer.convert_ids_to_tokens(target_ids[0])

        return [output_tokens, sequence_probability, sequence_rank]


    def get_sequence_probability_causal(self, sequence, target_sequence):
        self.load_causal_model()
        # Encode the combined input and target sequences
        combined_sequence = f"{sequence} {target_sequence}"
        input_ids = self.tokenizer.encode(combined_sequence, return_tensors='pt')

        # Calculate the logits for the encoded sequences
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Convert logits to probabilities
        probabilities = torch.softmax(logits, dim=-1)

        # Get token probabilities for each position in the target sequence
        target_ids = input_ids[0, -len(self.tokenizer.encode(target_sequence)):].squeeze()
        token_probabilities = probabilities[0, range(-len(target_ids), 0), target_ids]

        # Print the token probabilities
        for token, probability in zip(target_ids, token_probabilities):
            decoded_token = self.tokenizer.decode(token.unsqueeze(0))
            print(f"{decoded_token}: {probability.item()}")


    def compute_acc(self):
        self.load_seq2seq_model()
        test_data = self._load_json("./data_my/newsroom_test.json")
        acc = 0

        for thres in range(-30,-60,-2):
            count = 0
            for dp in tqdm(test_data):
                if not dp["text"] or not dp["summary"]: continue
                prob = self.get_sequence_probability(sequence=dp["text"], target_sequence=dp["summary"])
                if min(prob[1]) < thres:
                    predict = 0
                else:
                    predict = 1
                count += 1
                acc += predict == dp["label"]
            acc /= count
            print(f"With threshold of {thres}, the acc is {acc}")


def main():
    gen = Generation()
    gen.gen_prob()
    # gen.compute_acc()


if __name__ == "__main__":
    main()
