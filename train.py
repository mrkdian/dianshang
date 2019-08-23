import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import pandas as pd

from nltk import FreqDist

from pytorch_transformers import BertModel, load_tf_weights_in_bert, BertConfig, BertForPreTraining, BertTokenizer
from torchcrf import CRF

import json
import re

def test():
    train_label_df = pd.read_csv('TRAIN/Train_labels.csv', index_col='id')
    test_label_df = pd.read_csv('Result2.csv', header=None, index_col=0)
    test_review_df = pd.read_csv('TRAIN/TEST/Test_reviews.csv', index_col='id')

    aspect_dict = FreqDist()
    opinion_dict = FreqDist()
    a_o_diff_dict = FreqDist()

    for idx, row in train_label_df.iterrows():
        aspect = row['AspectTerms']
        opinion = row['OpinionTerms']
        if aspect != '_' and opinion != '_':
            diff = int(row['A_start']) - int(row['O_start'])
            a_o_diff_dict[diff] += 1
        if aspect != '_':
            aspect_dict[aspect] += 1
        if opinion != '_':
            opinion_dict[opinion] += 1
    
    for idx, row in test_label_df.iterrows():
        print(idx)
        review = test_review_df.loc[idx, 'Reviews']
        aspect = row[1]
        opinion = row[2]
        if aspect != '_':
            pattern = '.*' + aspect + '.*'
            for train_aspect in aspect_dict.keys():
                if train_aspect != aspect and re.match(pattern, train_aspect) and train_aspect in review:
                    print(train_aspect)
        if opinion != '_':
            pattern = '.*' + opinion + '.*'
            for train_opinion in opinion_dict.keys():
                if train_opinion != opinion and re.match(pattern, train_opinion) and train_opinion in review:
                    print(opinion, opinion_dict[opinion], train_opinion, opinion_dict[train_opinion], review)



    print(1)

if __name__ == '__main__':
    test()