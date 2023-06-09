#!/usr/bin/env python3
#
import sys, os, json
import pdb
from tqdm import tqdm

class SimplifyData(object):
    """docstring for SimplifyData"""
    def __init__(self, arg=None):
        super(SimplifyData, self).__init__()
        self.arg = arg
        self.data_dir = {
            "train": "/local-scratch1/data/qkun/semafor/semafor_data_collection/",
            "val"  : "/local-scratch1/data/qkun/semafor/Example_Probeset/3.3.4_MMA_Inconsistencies/Appeal To Fear/",
            "test" : "/local-scratch1/data/qkun/semafor/Example_Probeset/3.3.4_MMA_Inconsistencies/Minimization/"
        }
        self.save_dir = "./data/"
        

    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
            # return None
        with open(path) as df:
            data = json.loads(df.read())
        return data


    def _load_dir_json(self, dir_path=None, mode="train"):
        if dir_path is None or not os.path.exists(dir_path):
            raise IOError('Folder does not exist: %s' % dir_path)
        total_data = []
        if mode == "train":
            for subdir in tqdm(os.listdir(dir_path)):
                file_path = os.path.join(dir_path, subdir)
                data = self._load_json(os.path.join(dir_path, subdir, "aom.json"))
                total_data.append(data)
        else:
            for subdir in os.listdir(dir_path):
                for filename in os.listdir(os.path.join(dir_path, subdir)):
                    if os.path.isdir(os.path.join(dir_path, subdir, filename)):
                        data = self._load_json(os.path.join(dir_path, subdir, filename, "aom.json"))
                        total_data.append(data)
        return total_data


    def _save_json(self, data, path):
        with open(path, "w") as tf:
            json.dump(data, tf, indent=2)


    def save_simp_article(self, data, file_idx=0, mode="train"):
        save_name = f"articles_{file_idx}.json"
        folder_path = os.path.join(self.save_dir, mode)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        self._save_json(data, os.path.join(folder_path, save_name))


    def init_simple_article(self):
        new_article = {
            "title" : "",
            "content" : [],
            "uri" : "",
            "tags" : []
        }
        return new_article


    def simplify(self):
        for mode in ["train", "val", "test"]:
            # initilization
            data_simp, file_idx, miss_title = [], 0, 0
            data = self._load_dir_json(self.data_dir[mode], mode)
            for idx, article in tqdm(enumerate(data)):
                # initilization for each article
                article_simp = self.init_simple_article()
                article_simp["title"] = article["title"]
                article_simp["uri"] = article["uri"]
                article_simp["tags"] = article["tags"]
                if not article["title"]:
                    miss_title += 1
                    continue
                
                # extract context
                for sent in article["content"]:
                    if sent["Type"] == "Text":
                        article_simp["content"].extend(sent["Content"])
                    else:
                        for component in sent["Content"]:
                            if component["Type"] == "Text":
                                article_simp["content"].extend(component["Content"])
                data_simp.append(article_simp)

                if len(data_simp) == 5000 or idx == len(data)-1:
                    self.save_simp_article(data_simp, file_idx, mode)
                    data_simp = []
                    file_idx += 1
            print(f"Finish extracting and simplifying {mode} data, skip {miss_title}/{len(data)} articles without titles")


class SimplifyData4(SimplifyData):
    def __init__(self, arg=None):
        super(SimplifyData, self).__init__()
        self.data_dir = "./4.1.2"
    
    def simplify(self):
        data_simp = []
        for split in ["CallToAction", "Demoralize"]:
            for article in os.listdir(os.path.join(self.data_dir, split)):
                extract_dir = os.path.join(self.data_dir, split, article, "extract")
                article_simp = self.init_simple_article()
                article_simp["source"] = split
                twitter_posts = []
                for twitter in os.listdir(extract_dir):
                    data = self._load_json(os.path.join(extract_dir, twitter, "aom.json"))
                    # pdb.set_trace()
                    if not len(data["content"]) > 2: # it is a twitter post
                        twitter_posts.append(data["content"][1]["Content"][0]["Content"][0])
                    else:
                        # initilization for each article
                        article_simp["title"] = data["title"]
                        article_simp["uri"] = data["uri"]
                        article_simp["tags"] = data["tags"]
                        if not data["title"]:
                            miss_title += 1
                            continue
                        
                        # extract context
                        for sent in data["content"]:
                            if sent["Type"] == "Text":
                                article_simp["content"].extend(sent["Content"])
                            else:
                                for component in sent["Content"]:
                                    if component["Type"] == "Text":
                                        article_simp["content"].extend(component["Content"])
                        article_simp["content"] = [sent.strip() for sent in article_simp["content"]]
                        article_simp["content"] = " ".join(article_simp["content"])
                for twitter_post in twitter_posts:
                    data_point = {"twitter_post": twitter_post}
                    data_point.update(article_simp)
                    data_simp.append(data_point)
        self._save_json(data_simp, os.path.join(self.data_dir, "data_simplified.json"))


def main():
    # simplifydata = SimplifyData()
    # simplifydata.simplify()
    simplifydata = SimplifyData4()
    simplifydata.simplify()


if __name__ == '__main__':
    main()