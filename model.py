import torch
import torch.nn as nn
import torch.utils.data as torch_data
from torch.autograd import Variable
from nltk import FreqDist

import numpy as np
import time
import datetime
from tqdm import tqdm
import pandas as pd
import math
from pytorch_transformers import BertModel, BertTokenizer, BertConfig, BertForPreTraining, load_tf_weights_in_bert

from torchcrf import CRF

from sklearn.metrics import f1_score, precision_score, recall_score

import os
import pickle
import re
import json

UNK_TOKEN = 100

B_A = 1 # begin of aspect
I_A = 2
B_O = 3 # begin of opinion
I_O = 4

CATEGORY2ID = {
    '价格': 0,
    '使用体验': 1,
    '其他': 2,
    '功效': 3,
    '包装': 4,
    '尺寸': 5,
    '成分': 6,
    '整体': 7,
    '新鲜度': 8,
    '服务': 9,
    '气味': 10,
    '物流': 11,
    '真伪': 12
}
POLARITY2ID = {
    '正面': 0,
    '中性': 1,
    '负面': 2
}
def sub_list_index(list, sub_list):
    matches = []
    for i in range(len(list)):
        if list[i] == sub_list[0] and list[i: i+len(sub_list)] == sub_list:
            matches.append(i)
    return matches

def convert_tf_checkpoint_to_pytorch(tf_checkpoint_path, bert_config_file):
    # Initialise PyTorch model
    config = BertConfig.from_json_file(bert_config_file)
    print("Building PyTorch model from configuration: {}".format(str(config)))
    model = BertForPreTraining(config)

    # Load weights from tf checkpoint
    load_tf_weights_in_bert(model, config, tf_checkpoint_path)

    # Save pytorch-model
    # print("Save PyTorch model to {}".format(pytorch_dump_path))
    # torch.save(model.state_dict(), pytorch_dump_path)
    return model

def load_dataset(review_filepath, label_filepath, tokenizer, unk_upper_count=3):
    review_df = pd.read_csv(review_filepath, index_col='id')
    label_df = pd.read_csv(label_filepath, index_col='id')

    vocab = FreqDist()
    dataset = dict()
    for idx, row in review_df.iterrows():
        review = row['Reviews']
        tokens = tokenizer.encode(review)
        for token in tokens:
            vocab[token] += 1

        label = label_df.loc[idx]
        dataset[idx] = {
            'review': review,
            'label': []
        }
    
    for idx, row in label_df.iterrows():
        dataset[idx]['label'].append(list(row))

    known_token = set()
    for char, char_count in vocab.items():
        if char_count >= unk_upper_count:
            known_token.add(char)

    # total_token = 0
    # vocab = sorted(list(vocab.items()), key=lambda x: x[1])
    # vocab_freq = FreqDist()
    # vocab_acc_freq = [(0, 0)]
    # vocab_ratio = []
    # vocab_acc_ratio = [(0, 0)]
    # for char, count in vocab:
    #     vocab_freq[count] += 1
    #     total_token += count
    # for count, char_count in vocab_freq.items():
    #     vocab_freq[count] = char_count / len(vocab)
    #     vocab_acc_freq.append((count, vocab_acc_freq[-1][1] + vocab_freq[count]))
    #     vocab_ratio.append((count, count * char_count / total_token))
    # for count, ratio in vocab_ratio:
    #     vocab_acc_ratio.append((count, vocab_acc_ratio[-1][1] + ratio))
    return dataset, known_token

def split_dataset(dataset, shuffle_idx_file, ratio=0.6):
    if not os.path.exists(shuffle_idx_file):
        idx = list(dataset.keys())
        np.random.shuffle(idx)
        pickle.dump(idx, open(shuffle_idx_file, mode='wb'))
    else:
        idx = pickle.load(open(shuffle_idx_file, mode='rb'))

    train_dataset = list(filter(lambda x: x[0] in idx[:int(len(idx) * ratio)], dataset.items()))
    validate_dataset = list(filter(lambda x: x[0] in idx[int(len(idx) * ratio):], dataset.items()))

    return train_dataset, validate_dataset

class Dataset(torch_data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

class collate_fn:
    def __init__(self, tokenizer, known_token=None):
        self.tokenizer = tokenizer
        self.known_token = known_token
    
    def __call__(self, batch_data):
        batch_size = len(batch_data)
        batch_X = []
        len_X = []
        seq_target = []
        mask = []
        aspect_idx = []
        opinion_idx = []
        match_target = []
        single_aspect_category_target = []
        single_opinion_category_target = []
        single_aspect_polarity_target = []
        single_opinion_polarity_target = []
        cross_category_target = []
        cross_polarity_target = []

        for idx, data in batch_data:
            review = data['review']
            labels = data['label']

            x = self.tokenizer.encode('[CLS]' + review)
            # batch_X.append(torch.tensor(x))
            len_X.append(len(x))
            mask.append(torch.ones(len(x), dtype=torch.uint8))
            _seq_target = torch.zeros(len(x), dtype=torch.long)
            _seq_target[0] = -100 #  ignore [CLS] token

            c_p_label = [] # category and polarity label
            success_match = []
            _aspect_idx = []
            _opinion_idx = []
            for label in labels:
                #Aspect term
                if label[0] != '_':
                    label[1] = int(label[1])
                    label[2] = int(label[2])

                    # foolish method to ensure the label position
                    temp = self.tokenizer.encode(review[:label[1]] + '[PAD]' + review[label[1]: label[2]] + '[PAD]')
                    p = temp.index(0)
                    _seq_target[p + 1] = B_A
                    _seq_target[p + 2: len(temp) - 1] = I_A
                    _aspect_idx.append(list(range(p + 1, len(temp) - 1)))
                    assert self.tokenizer.decode(x[p + 1: len(temp) - 1]).replace(' ', '') == label[0].lower()
                #Opinion term
                if label[3] != '_':
                    label[4] = int(label[4])
                    label[5] = int(label[5])

                    temp = self.tokenizer.encode(review[:label[4]] + '[PAD]' + review[label[4]: label[5]] + '[PAD]')
                    p = temp.index(0)
                    _seq_target[p + 1] = B_O
                    _seq_target[p + 2: len(temp) - 1] = I_O
                    _opinion_idx.append(list(range(p + 1, len(temp) - 1)))
                    assert self.tokenizer.decode(x[p + 1: len(temp) - 1]).replace(' ', '') == label[3].lower()

                if label[0] != '_' and label[3] != '_':
                    success_match.append((len(_aspect_idx) - 1, len(_opinion_idx) - 1))
                    c_p_label.append((len(_aspect_idx) - 1, len(_opinion_idx) - 1, CATEGORY2ID[label[6]], POLARITY2ID[label[7]]))
                elif label[0] != '_':
                    c_p_label.append((len(_aspect_idx) - 1, None, CATEGORY2ID[label[6]], POLARITY2ID[label[7]]))
                elif label[3] != '_':
                    c_p_label.append((None, len(_opinion_idx) - 1, CATEGORY2ID[label[6]], POLARITY2ID[label[7]]))

                assert not (label[0] == '_' and label[3] == '_')

            if self.known_token:
                for p, token_idx in enumerate(x):
                    if token_idx not in self.known_token:
                        x[p] = UNK_TOKEN
            batch_X.append(torch.tensor(x))
            
            _match_target = torch.zeros(len(_aspect_idx), len(_opinion_idx), dtype=torch.float) # binary cross entropy
            for a_id, o_id in success_match:
                _match_target[a_id, o_id] = 1
            
            _single_aspect_category_target = torch.empty(len(_aspect_idx), dtype=torch.long).fill_(-100)
            _single_opinion_category_target = torch.empty(len(_opinion_idx), dtype=torch.long).fill_(-100)
            _single_aspect_polarity_target = torch.empty(len(_aspect_idx), dtype=torch.long).fill_(-100)
            _single_opinion_polarity_target = torch.empty(len(_opinion_idx), dtype=torch.long).fill_(-100)
            _cross_category_target = torch.empty(len(_aspect_idx), len(_opinion_idx), dtype=torch.long).fill_(-100)
            _cross_polarity_target = torch.empty(len(_aspect_idx), len(_opinion_idx), dtype=torch.long).fill_(-100)

            for a_id, o_id, c_label, p_label in c_p_label:
                if a_id is not None and o_id is not None:
                    _cross_category_target[a_id, o_id] = c_label
                    _cross_polarity_target[a_id, o_id] = p_label
                elif a_id is not None:
                    _single_aspect_category_target[a_id] = c_label
                    _single_aspect_polarity_target[a_id] = p_label
                elif o_id is not None:
                    _single_opinion_category_target[o_id] = c_label
                    _single_opinion_polarity_target[o_id] = p_label

            match_target.append(_match_target)
            seq_target.append(_seq_target)
            aspect_idx.append(_aspect_idx)
            opinion_idx.append(_opinion_idx)
            single_aspect_category_target.append(_single_aspect_category_target)
            single_opinion_category_target.append(_single_opinion_category_target)
            single_aspect_polarity_target.append(_single_aspect_polarity_target)
            single_opinion_polarity_target.append(_single_opinion_polarity_target)
            cross_category_target.append(_cross_category_target)
            cross_polarity_target.append(_cross_polarity_target)


        max_len = max(len_X)
        seq_target = nn.utils.rnn.pad_sequence(seq_target, batch_first=True, padding_value=-100)
        batch_X = nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0)
        mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
        assert batch_X.shape[1] == max_len
        
        targets = (seq_target, match_target, single_aspect_category_target, single_opinion_category_target, cross_category_target,\
            single_aspect_polarity_target, single_opinion_polarity_target, cross_polarity_target)

        return batch_X, len_X, mask, (aspect_idx, opinion_idx), targets

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def get_pe(self, p):
        return self.pe[:, p].squeeze()

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class SeqEncoder(nn.Module):
    pass

class Model(nn.Module):
    def __init__(self, bert):
        super().__init__()
        self.bert = bert
        self.emb_size = bert.config.hidden_size
        # self.general_lstm = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size,\
        #     batch_first=True)

        self.crf = CRF(num_tags=5, batch_first=True)
        # self.positional_encoding = PositionalEncoding(self.emb_size, 0.1, max_len=512)

        self.seq_labeling = nn.Linear(self.emb_size, 5)

        # self.aspect_encoder = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)
        # self.opinion_encoder = nn.LSTM(input_size=self.emb_size, hidden_size=self.emb_size, batch_first=True)

        # self.lstm_a_att_q = nn.Linear(self.emb_size, 1) # query
        # self.lstm_o_att_q = nn.Linear(self.emb_size, 1)
        # self.lstm_a_att_k = nn.Linear(self.emb_size, self.emb_size) # key
        # self.lstm_o_att_k = nn.Linear(self.emb_size, self.emb_size)
        # self.lstm_a_att_v = nn.Linear(self.emb_size, self.emb_size) # value
        # self.lstm_o_att_v = nn.Linear(self.emb_size, self.emb_size)

        self.aspect_att_query = nn.Linear(self.emb_size, 1)
        self.opinion_att_query = nn.Linear(self.emb_size, 1)
        self.aspect_att_key = nn.Linear(self.emb_size, self.emb_size)
        self.opinion_att_key = nn.Linear(self.emb_size, self.emb_size)
        self.aspect_att_val = nn.Linear(self.emb_size, self.emb_size)
        self.opinion_att_val = nn.Linear(self.emb_size, self.emb_size)

        self.match = nn.Sequential(
            nn.Linear(self.emb_size * 2, self.emb_size),
            nn.SELU(),
            nn.Linear(self.emb_size, self.emb_size // 2),
            nn.SELU(),
            nn.Linear(self.emb_size // 2, 1)
        )

        # self.aspect_match = nn.Linear(self.emb_size, 2)
        # self.opinion_match = nn.Linear(self.emb_size, 2)

        # self.aspect_match = nn.Sequential(
        #     nn.Linear(self.emb_size, self.emb_size // 2),
        #     nn.SELU(),
        #     nn.Linear(self.emb_size // 2, 2)
        # )
        # self.opinion_match = nn.Sequential(
        #     nn.Linear(self.emb_size, self.emb_size // 2),
        #     nn.SELU(),
        #     nn.Linear(self.emb_size // 2, 2)
        # )

        self.aspect_category = nn.Linear(self.emb_size, len(CATEGORY2ID.keys()))
        self.opinion_category = nn.Linear(self.emb_size, len(CATEGORY2ID.keys()))
        self.aspect_polarity = nn.Linear(self.emb_size, len(POLARITY2ID.keys()))
        self.opinion_polarity = nn.Linear(self.emb_size, len(POLARITY2ID.keys()))

        self.cross_entropy = nn.CrossEntropyLoss()
    def seq_gather(self):
        pass

    def collate_aspect_opinion(self, seq_score, len_X):
        batch_size = seq_score.shape[0]
        aspect_idx = []
        opinion_idx = []
        seq_target = torch.argmax(seq_score, dim=-1).cpu().detach().numpy()
        for b in range(batch_size):
            _aspect_idx = []
            _opinion_idx = []
            current = 0 # 0: other, 1: aspect, 2: opinion
            current_idx = []
            for p in range(len_X[b]):
                if seq_target[b, p] == 0:
                    if current == 0:
                        continue
                    elif current == 1:
                        _aspect_idx.append(current_idx)
                    elif current == 2:
                        _opinion_idx.append(current_idx)
                    current = 0
                    current_idx = []
                elif seq_target[b, p] == 1:
                    if current == 1:
                        _aspect_idx.append(current_idx)
                        current_idx = []
                    elif current == 2:
                        _opinion_idx.append(current_idx)
                        current_idx = []
                    current = 1
                    current_idx.append(p)
                elif seq_target[b, p] == 2:
                    if current == 0:
                        continue
                    elif current == 1:
                        current_idx.append(p)
                    elif current == 2:
                        _opinion_idx.append(current_idx)
                        current_idx = []
                        current = 0
                elif seq_target[b, p] == 3:
                    if current == 1:
                        _aspect_idx.append(current_idx)
                        current_idx = []
                    elif current == 2:
                        _opinion_idx.append(current_idx)
                        current_idx = []
                    current = 2
                    current_idx.append(p)
                elif seq_target[b, p] == 4:
                    if current == 0:
                        continue
                    elif current == 1:
                        _aspect_idx.append(current_idx)
                        current_idx = []
                        current = 0
                    elif current == 2:
                        current_idx.append(p)
            if current == 1:
                _aspect_idx.append(current_idx)
            elif current == 2:
                _opinion_idx.append(current_idx)
            aspect_idx.append(_aspect_idx)
            opinion_idx.append(_opinion_idx)
        return (aspect_idx, opinion_idx)

    def forward(self, batch_X, len_X, mask, gather_idx=None):
        batch_size = batch_X.shape[0]
        device = batch_X.device

        bert_emb, _ = self.bert(batch_X, attention_mask=mask)
        # g_lstm_emb, _ = self.general_lstm(bert_emb)
        seq_score = self.seq_labeling(g_lstm_emb)

        if gather_idx is None:
            gather_idx = self.collate_aspect_opinion(seq_score, len_X)

        match_score = []
        single_aspect_category_score = []
        single_opinion_category_score = []
        single_aspect_polarity_score = []
        single_opinion_polarity_score = []
        cross_category_score = []
        cross_polarity_score = []

        aspect_idx, opinion_idx = gather_idx
        for b in range(batch_size):
            _aspect_idx = aspect_idx[b]
            _opinion_idx = opinion_idx[b]

            # lstm_aspect_out = []
            # lstm_opinion_out = []
            bert_aspect_out = []
            bert_opinion_out = []
            _single_aspect_category_score = []
            _single_opinion_category_score = []
            _single_aspect_polarity_score = []
            _single_opinion_polarity_score = []

            for idx in _aspect_idx:
                bert_aspect_emb = torch.unsqueeze(bert_emb[b, idx], 0)
                # lstm_aspect_emb = torch.unsqueeze(g_lstm_emb[b, idx], 0)
                # _, (a_h_out, a_c_cout) = self.aspect_encoder(aspect_emb)
                # a_h_out = torch.unsqueeze(torch.sum(aspect_emb, dim=1), 1)
                
                # attention mechanism from bert
                key = self.aspect_att_key(bert_aspect_emb)
                val = self.aspect_att_val(bert_aspect_emb)
                att = self.aspect_att_query(key)
                val = val * att
                bert_a_out = torch.unsqueeze(torch.sum(val, dim=1), 1)

                # attention mechanism from general lstm
                # key = self.lstm_a_att_k(lstm_aspect_emb)
                # val = self.lstm_a_att_v(lstm_aspect_emb)
                # att = self.lstm_a_att_q(key)
                # val = val * att
                # lstm_a_out = torch.unsqueeze(torch.sum(val, dim=1), 1)

                s_a_c_out = torch.squeeze(self.aspect_category(bert_a_out))
                s_a_p_out = torch.squeeze(self.aspect_polarity(bert_a_out))

                # lstm_aspect_out.append(lstm_a_out)
                bert_aspect_out.append(bert_a_out)
                _single_aspect_category_score.append(s_a_c_out)
                _single_aspect_polarity_score.append(s_a_p_out)
            for idx in _opinion_idx:
                bert_opinion_emb = torch.unsqueeze(bert_emb[b, idx], 0)
                # lstm_opinion_emb = torch.unsqueeze(g_lstm_emb[b, idx], 0)
                # _, (o_h_out, o_c_out) = self.opinion_encoder(opinion_emb)
                # o_h_out = torch.unsqueeze(torch.sum(opinion_emb, dim=1), 1)

                # attention mechanism from bert
                key = self.opinion_att_key(bert_opinion_emb)
                val = self.opinion_att_val(bert_opinion_emb)
                att = self.opinion_att_query(key)
                val = val * att
                bert_o_out = torch.unsqueeze(torch.sum(val, dim=1), 1)

                # attention mechanism from general lstm
                # key = self.lstm_o_att_k(lstm_opinion_emb)
                # val = self.lstm_o_att_v(lstm_opinion_emb)
                # att = self.lstm_o_att_q(key)
                # val = val * att
                # lstm_o_out = torch.unsqueeze(torch.sum(val, dim=1), 1)

                s_o_c_out = torch.squeeze(self.opinion_category(bert_o_out))
                s_o_p_out = torch.squeeze(self.opinion_polarity(bert_o_out))

                # lstm_opinion_out.append(lstm_o_out)
                bert_opinion_out.append(bert_o_out)
                _single_opinion_category_score.append(s_o_c_out)
                _single_opinion_polarity_score.append(s_o_p_out)

            # assert len(bert_aspect_out) == len(lstm_aspect_out)
            # assert len(bert_opinion_out) == len(lstm_opinion_out)
            _cross_category_score = torch.empty(len(bert_aspect_out), len(bert_opinion_out), len(CATEGORY2ID.keys()), device=device)
            _cross_polarity_score = torch.empty(len(bert_aspect_out), len(bert_opinion_out), len(POLARITY2ID.keys()), device=device)
            _match_score = torch.empty(len(bert_aspect_out), len(bert_opinion_out), device=device)
            for i in range(len(bert_aspect_out)):
                for j in range(len(bert_opinion_out)):
                    bert_a_out = bert_aspect_out[i]
                    # lstm_a_out = lstm_aspect_out[i]
                    # a_p = _aspect_idx[i][0]
                    bert_o_out = bert_opinion_out[j]
                    # lstm_o_out = lstm_opinion_out[j]
                    # o_p = _opinion_idx[j][0]

                    _match_score[i, j] = self.match(torch.cat((bert_a_out, bert_o_out), dim=-1))
                    
                    # _match_score[i, j] = self.aspect_match(lstm_a_out) + self.opinion_match(lstm_o_out)
                    # _match_score[i, j] = self.aspect_match(a_h_out.squeeze() + self.positional_encoding.get_pe(a_p)) + \
                    #     self.opinion_match(o_h_out.squeeze() + self.positional_encoding.get_pe(o_p))

                    _cross_category_score[i, j] = _single_aspect_category_score[i] + _single_opinion_category_score[j]
                    _cross_polarity_score[i, j] = _single_aspect_polarity_score[i] + _single_opinion_polarity_score[j]
            
            match_score.append(_match_score)
            single_aspect_category_score.append(_single_aspect_category_score)
            single_opinion_category_score.append(_single_opinion_category_score)
            single_aspect_polarity_score.append(_single_aspect_polarity_score)
            single_opinion_polarity_score.append(_single_opinion_polarity_score)
            cross_category_score.append(_cross_category_score)
            cross_polarity_score.append(_cross_polarity_score)

        return (seq_score, match_score, single_aspect_category_score, single_opinion_category_score, cross_category_score,\
            single_aspect_polarity_score, single_opinion_polarity_score, cross_polarity_score), gather_idx

    def loss(self, scores, targets, X_mask):
        (seq_score, match_score, single_aspect_category_score, single_opinion_category_score, cross_category_score,\
            single_aspect_polarity_score, single_opinion_polarity_score, cross_polarity_score) = scores
        (seq_target, match_target, single_aspect_category_target, single_opinion_category_target, cross_category_target,\
            single_aspect_polarity_target, single_opinion_polarity_target, cross_polarity_target) = targets
        device = seq_score.device

        seq_target = torch.empty_like(seq_target).copy_(seq_target)
        seq_target = seq_target.to(device)
        seq_target[seq_target == -100] = 0

        seq_labeling_loss = -self.crf(seq_score[:, 1:], seq_target[:, 1:], mask=X_mask[:, 1:], reduction='mean')
        # seq_labeling_loss = self.cross_entropy(seq_score.view(-1, seq_score.shape[-1]), seq_target.view(-1))

        category_loss = 0
        category_count = 0
        polarity_loss = 0
        polarity_count = 0
        match_loss = 0
        match_count = 0
        for b in range(len(match_target)):
            if match_target[b].numel() > 0:
                # calculate the probility
                match_prob = torch.empty_like(match_score[b])
                for i in range(match_score[b].shape[0]):
                    for j in range(match_score[b].shape[1]):
                          score = torch.cat((match_score[b][i], match_score[b][:i, j], match_score[b][i + 1:, j]))
                          # assert len(score) == match_score[b].shape[0] + match_score[b].shape[1] - 1
                          match_prob[i, j] = nn.functional.softmax(score, dim=0)[j]

                match_count += match_target[b].numel()
                match_target[b] = match_target[b].to(device)
                match_loss += nn.functional.binary_cross_entropy(match_prob.view(-1),\
                    match_target[b].view(-1), reduction='sum')
                # match_loss += nn.functional.cross_entropy(match_score[b].view(-1, match_score[b].shape[-1]),\
                #     match_target[b].view(-1), reduction='sum')

            _category_loss = 0
            _category_count = 0
            if len(single_aspect_category_score[b]) > 0:
                single_aspect_category_target[b] = single_aspect_category_target[b].to(device)
                _single_aspect_category_score = torch.stack(single_aspect_category_score[b], dim=0)
                _category_count += torch.sum(single_aspect_category_target[b] != -100)
                _category_loss += nn.functional.cross_entropy(_single_aspect_category_score, single_aspect_category_target[b], reduction='sum')
            if len(single_opinion_category_score[b]) > 0:
                single_opinion_category_target[b] = single_opinion_category_target[b].to(device)
                _single_opinion_category_score = torch.stack(single_opinion_category_score[b], dim=0)
                _category_count += torch.sum(single_opinion_category_target[b] != -100)
                _category_loss += nn.functional.cross_entropy(_single_opinion_category_score, single_opinion_category_target[b], reduction='sum')
            if cross_category_score[b].numel() > 0:
                cross_category_target[b] = cross_category_target[b].to(device)
                _category_loss += nn.functional.cross_entropy(cross_category_score[b].view(-1, cross_category_score[b].shape[-1]),\
                    cross_category_target[b].view(-1), reduction='sum')
                _category_count += torch.sum(cross_category_target[b] != -100)

            _polarity_loss = 0
            _polarity_count = 0
            if len(single_aspect_polarity_score[b]) > 0:
                single_aspect_polarity_target[b] = single_aspect_polarity_target[b].to(device)
                _single_aspect_polarity_score = torch.stack(single_aspect_polarity_score[b], dim=0)
                _polarity_count += torch.sum(single_aspect_polarity_target[b] != -100)
                _polarity_loss += nn.functional.cross_entropy(_single_aspect_polarity_score, single_aspect_polarity_target[b], reduction='sum')
            if len(single_opinion_polarity_score[b]) > 0:
                single_opinion_polarity_target[b] = single_opinion_polarity_target[b].to(device)
                _single_opinion_polarity_score = torch.stack(single_opinion_polarity_score[b], dim=0)
                _polarity_count += torch.sum(single_opinion_polarity_target[b] != -100)
                _polarity_loss += nn.functional.cross_entropy(_single_opinion_polarity_score, single_opinion_polarity_target[b], reduction='sum')
            if cross_polarity_score[b].numel() > 0:
                cross_polarity_target[b] = cross_polarity_target[b].to(device)
                _polarity_loss += nn.functional.cross_entropy(cross_polarity_score[b].view(-1, cross_polarity_score[b].shape[-1]), \
                    cross_polarity_target[b].view(-1), reduction='sum')
                _polarity_count += torch.sum(cross_polarity_target[b] != -100)
            
            category_loss += _category_loss
            category_count += _category_count
            polarity_loss += _polarity_loss
            polarity_count += _polarity_count

        if match_count > 0:
            match_loss = match_loss / match_count

        assert category_count == polarity_count
        category_loss = category_loss / category_count
        polarity_loss = polarity_loss / polarity_count
            
        total_loss = seq_labeling_loss + match_loss + category_loss + polarity_loss
        return (total_loss, seq_labeling_loss, match_loss, category_loss, polarity_loss)
    
    def infer(self, scores, X_mask):
        (seq_score, match_score, single_aspect_category_score, single_opinion_category_score, cross_category_score,\
            single_aspect_polarity_score, single_opinion_polarity_score, cross_polarity_score) = scores
        device = seq_score.device

        # seq_target = torch.argmax(seq_score, dim=-1)
        seq_target = self.crf.decode(seq_score[:, 1:], X_mask[:, 1:])
        for i, _seq_target in enumerate(seq_target):
            _seq_target.insert(0, -100)
            seq_target[i] = torch.tensor(_seq_target)
        seq_target = nn.utils.rnn.pad_sequence(seq_target, batch_first=True, padding_value=-100)

        match_target = []
        single_aspect_category_target = []
        single_opinion_category_target = []
        cross_category_target = []
        single_aspect_polarity_target = []
        single_opinion_polarity_target = []
        cross_polarity_target = []
        for b in range(len(match_score)):
            if match_score[b].numel() < 1:
                match_target.append(None)
            else:
                _match_target = torch.empty_like(match_score[b], dtype=torch.long)
                for i in range(match_score[b].shape[0]):
                    for j in range(match_score[b].shape[1]):
                        score = torch.cat((match_score[b][i], match_score[b][:i, j], match_score[b][i + 1:, j]))
                        if torch.argmax(score) == j:
                            _match_target[i, j] = 1
                        else:
                            _match_target[i, j] = 0
                match_target.append(_match_target)
                # match_target.append(torch.argmax(match_score[b], dim=-1))
            #category
            if len(single_aspect_category_score[b]) > 0:
                _single_aspect_category_score = torch.stack(single_aspect_category_score[b], dim=0)
                single_aspect_category_target.append(torch.argmax(_single_aspect_category_score, dim=-1))
            else:
                single_aspect_category_target.append(None)
            if len(single_opinion_category_score[b]) > 0:
                _single_opinion_category_score = torch.stack(single_opinion_category_score[b], dim=0)
                single_opinion_category_target.append(torch.argmax(_single_opinion_category_score, dim=-1))
            else:
                single_opinion_category_target.append(None)
            if cross_category_score[b].numel() > 0:
                cross_category_target.append(torch.argmax(cross_category_score[b], dim=-1))
            else:
                cross_category_target.append(None)
            #polarity
            if len(single_aspect_polarity_score[b]) > 0:
                _single_aspect_polarity_score = torch.stack(single_aspect_polarity_score[b], dim=0)
                single_aspect_polarity_target.append(torch.argmax(_single_aspect_polarity_score, dim=-1))
            else:
                single_aspect_polarity_target.append(None)

            if len(single_opinion_polarity_score[b]) > 0:
                _single_opinion_polarity_score = torch.stack(single_opinion_polarity_score[b], dim=0)
                single_opinion_polarity_target.append(torch.argmax(_single_opinion_polarity_score, dim=-1))
            else:
                single_opinion_polarity_target.append(None)

            if cross_polarity_score[b].numel() > 0:
                cross_polarity_target.append(torch.argmax(cross_polarity_score[b], dim=-1))
            else:
                cross_polarity_target.append(None)

        return (seq_target, match_target, single_aspect_category_target, single_opinion_category_target, cross_category_target,\
            single_aspect_polarity_target, single_opinion_polarity_target, cross_polarity_target)
    

def seq_f1(pred_targets, gt_targets, average=None):
    mask = (gt_targets != -100)
    pred_targets = pred_targets[mask]
    gt_targets = gt_targets[mask]
    seq_metric = f1_score(gt_targets, pred_targets, average=average)
    return seq_metric


def test():
    # torch.autograd.set_detect_anomaly(True)
    load_save_model = False
    lr = 1e-5
    batch_size = 4
    gpu = True
    torch.manual_seed(0)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')

    tokenizer = BertTokenizer(vocab_file='publish/vocab.txt', max_len=512)
    dataset, known_token = load_dataset('TRAIN/Train_reviews.csv', 'TRAIN/Train_labels.csv', tokenizer)
    train_dataset, validate_dataset = split_dataset(dataset, 'TRAIN/shuffle.idx', 0.97)

    bert_pretraining = convert_tf_checkpoint_to_pytorch('./publish/bert_model.ckpt', './publish/bert_config.json')
    model = Model(bert_pretraining.bert)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='bert-base-chinese')
    train_dataset = Dataset(train_dataset)
    train_dataloader = torch_data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer)
    )
    validate_dataset = Dataset(validate_dataset)
    validate_dataloader = torch_data.DataLoader(
        dataset=validate_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn(tokenizer)
    )

    model = model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

    statistic = {
        'best_f1': -100,
        'best_f1_epoch': None,
        'best_match_f1': -100,
        'best_match_epoch': None,
        'epoch_detail': []
    }

    if load_save_model:
        model.load_state_dict(torch.load('./save_model/best.model'))

    for epoch in range(15):
        print(str(epoch) + '------------------------------------------------------------------')
        accum_total_loss = 0
        accum_seq_labeling_loss = 0
        accum_match_loss = 0
        accum_category_loss = 0
        accum_polarity_loss = 0

        model.train()
        pbar = tqdm()
        try:
            for step, (batch_X, len_X, mask, gather_idx, targets) in enumerate(train_dataloader):
                batch_X = batch_X.to(device)
                mask = mask.to(device)
                # tokenizer.decode(list(batch_X[0].cpu().numpy())).replace(' ', '')
                scores, gather_idx = model(batch_X, len_X, mask, gather_idx)
                loss = model.loss(scores, targets, mask)

                optimizer.zero_grad()
                loss[0].backward()
                optimizer.step()

                accum_total_loss += loss[0].cpu().detach().numpy()
                accum_seq_labeling_loss += loss[1].cpu().detach().numpy()
                if type(loss[2]) is not int:
                    accum_match_loss += loss[2].cpu().detach().numpy()
                accum_category_loss += loss[3].cpu().detach().numpy()
                accum_polarity_loss += loss[4].cpu().detach().numpy()

                pbar.update(batch_size)
                pbar.set_description('step: %d, total loss: %f, seq loss: %f, match loss: %f, category loss: %f, polarity loss: %f' % \
                    (step, accum_total_loss / (step + 1), accum_seq_labeling_loss / (step + 1), accum_match_loss / (step + 1),\
                        accum_category_loss / (step + 1), accum_polarity_loss / (step + 1)))
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()
        optimizer.zero_grad()
        loss_statistic = {
            'total_loss': accum_total_loss / (step + 1),
            'seq_loss': accum_seq_labeling_loss / (step + 1),
            'match_loss': accum_match_loss / (step + 1),
            'category_loss': accum_category_loss / (step + 1),
            'polarity_loss': accum_polarity_loss / (step + 1)
        }
        
        
        model.eval()
        total_gt_seq_target = []
        total_gt_match_target = []
        total_gt_single_aspect_category_target = []
        total_gt_single_opinion_category_target = []
        total_gt_cross_category_target = []
        total_gt_single_aspect_polarity_target = []
        total_gt_single_opinion_polarity_target = []
        total_gt_cross_polarity_target = []

        total_pred_seq_target = []
        total_pred_match_target = []
        total_pred_single_aspect_category_target = []
        total_pred_single_opinion_category_target = []
        total_pred_cross_category_target = []
        total_pred_single_aspect_polarity_target = []
        total_pred_single_opinion_polarity_target = []
        total_pred_cross_polarity_target = []
        pbar = tqdm()
        try:
            for step, (batch_X, len_X, mask, gather_idx, targets) in enumerate(validate_dataloader):
                batch_X = batch_X.to(device)
                mask = mask.to(device)

                scores, gather_idx = model(batch_X, len_X, mask, gather_idx)

                (pred_seq_target, pred_match_target, pred_single_aspect_category_target, pred_single_opinion_category_target,\
                    pred_cross_category_target, pred_single_aspect_polarity_target, pred_single_opinion_polarity_target,\
                        pred_cross_polarity_target) = model.infer(scores, mask)

                (seq_target, match_target, single_aspect_category_target, single_opinion_category_target, cross_category_target,\
                    single_aspect_polarity_target, single_opinion_polarity_target, cross_polarity_target) = targets

                total_pred_seq_target.append(pred_seq_target.view(-1).cpu().detach().numpy())
                total_gt_seq_target.append(seq_target.view(-1).cpu().detach().numpy())

                for b in range(len(pred_match_target)):
                    if pred_match_target[b] is not None:
                        assert match_target[b].numel() != 0
                        total_pred_match_target.append(pred_match_target[b].view(-1).cpu().detach().numpy())
                        total_gt_match_target.append(match_target[b].view(-1).cpu().detach().numpy())

                    if pred_single_aspect_category_target[b] is not None:
                        total_pred_single_aspect_category_target.append(pred_single_aspect_category_target[b].cpu().detach().numpy())
                        total_gt_single_aspect_category_target.append(single_aspect_category_target[b].cpu().detach().numpy())
                    if pred_single_opinion_category_target[b] is not None:
                        total_pred_single_opinion_category_target.append(pred_single_opinion_category_target[b].cpu().detach().numpy())
                        total_gt_single_opinion_category_target.append(single_opinion_category_target[b].cpu().detach().numpy())
                    if pred_cross_category_target[b] is not None:
                        total_pred_cross_category_target.append(pred_cross_category_target[b].view(-1).cpu().detach().numpy())
                        total_gt_cross_category_target.append(cross_category_target[b].view(-1).cpu().detach().numpy())
                    if pred_single_aspect_polarity_target[b] is not None:
                        total_pred_single_aspect_polarity_target.append(pred_single_aspect_polarity_target[b].cpu().detach().numpy())
                        total_gt_single_aspect_polarity_target.append(single_aspect_polarity_target[b].cpu().detach().numpy())
                    if pred_single_opinion_polarity_target[b] is not None:
                        total_pred_single_opinion_polarity_target.append(pred_single_opinion_polarity_target[b].cpu().detach().numpy())
                        total_gt_single_opinion_polarity_target.append(single_opinion_polarity_target[b].cpu().detach().numpy())
                    if pred_cross_polarity_target[b] is not None:
                        total_pred_cross_polarity_target.append(pred_cross_polarity_target[b].view(-1).cpu().detach().numpy())
                        total_gt_cross_polarity_target.append(cross_polarity_target[b].view(-1).cpu().detach().numpy())

                pbar.update(batch_size)
                pbar.set_description('step: %d' % step)
        except KeyboardInterrupt:
            pbar.close()
            raise
        pbar.close()

        total_gt_seq_target = np.concatenate(total_gt_seq_target)
        total_gt_match_target = np.concatenate(total_gt_match_target)
        total_gt_single_aspect_category_target = np.concatenate(total_gt_single_aspect_category_target)
        total_gt_single_opinion_category_target = np.concatenate(total_gt_single_opinion_category_target)
        total_gt_cross_category_target = np.concatenate(total_gt_cross_category_target)
        total_gt_single_aspect_polarity_target = np.concatenate(total_gt_single_aspect_polarity_target)
        total_gt_single_opinion_polarity_target = np.concatenate(total_gt_single_opinion_polarity_target)
        total_gt_cross_polarity_target = np.concatenate(total_gt_cross_polarity_target)

        total_pred_seq_target = np.concatenate(total_pred_seq_target)
        total_pred_match_target = np.concatenate(total_pred_match_target)
        total_pred_single_aspect_category_target = np.concatenate(total_pred_single_aspect_category_target)
        total_pred_single_opinion_category_target = np.concatenate(total_pred_single_opinion_category_target)
        total_pred_cross_category_target = np.concatenate(total_pred_cross_category_target)
        total_pred_single_aspect_polarity_target = np.concatenate(total_pred_single_aspect_polarity_target)
        total_pred_single_opinion_polarity_target = np.concatenate(total_pred_single_opinion_polarity_target)
        total_pred_cross_polarity_target = np.concatenate(total_pred_cross_polarity_target)

        total_gt_category_target = np.concatenate((total_gt_single_aspect_category_target, total_gt_single_opinion_category_target,\
            total_gt_cross_category_target))
        total_pred_category_target = np.concatenate((total_pred_single_aspect_category_target, total_pred_single_opinion_category_target,\
            total_pred_cross_category_target))
        total_gt_polarity_target = np.concatenate((total_gt_single_aspect_polarity_target, total_gt_single_opinion_polarity_target,\
            total_gt_cross_polarity_target))
        total_pred_polarity_target = np.concatenate((total_pred_single_aspect_polarity_target, total_gt_single_opinion_polarity_target,\
            total_pred_cross_polarity_target))

        seq_metric = seq_f1(total_pred_seq_target, total_gt_seq_target)
        match_f1 = f1_score(total_gt_match_target, total_pred_match_target)
        match_p = precision_score(total_gt_match_target, total_pred_match_target)
        match_r = recall_score(total_gt_match_target, total_pred_match_target)
        category_f1 = seq_f1(total_pred_category_target, total_gt_category_target, 'macro')
        polarity_f1 = seq_f1(total_pred_polarity_target, total_gt_polarity_target, 'macro')
        print('Others: %f, B_A: %f, I_A: %f, B_O: %f, I_O: %f, ' % tuple(seq_metric), end='')
        
        print('match: %f, ' % match_f1, end='')
        print('match p: %f, ' % match_p, end='')
        print('match r: %f, ' % match_r, end='')
        print('category: %f, ' % category_f1, end='')
        print('polarity: %f, ' % polarity_f1, end='')

        epoch_statistic = {
            'seq_metric': tuple(seq_metric),
            'seq':  'Others: %f, B_A: %f, I_A: %f, B_O: %f, I_O: %f, ' % tuple(seq_metric),
            'match': match_f1,
            'match_p': match_p,
            'match_r': match_r,
            'category:': category_f1,
            'polarity': polarity_f1,
            'loss': loss_statistic
        }
        avg_f1 = (np.sum(seq_metric) + match_f1 + category_f1 + polarity_f1) / 8
        print('avg: %f' % avg_f1)
        if avg_f1 > statistic['best_f1']:
            statistic['best_f1'] = avg_f1
            statistic['best_f1_epoch'] = epoch
            torch.save(model.state_dict(), 'save_model/best.model')
        if match_f1 > statistic['best_match_f1']:
            statistic['best_match_f1'] = match_f1
            statistic['best_match_epoch'] = epoch
            torch.save(model.state_dict(), 'save_model/best_match.model')
        statistic['epoch_detail'].append(epoch_statistic)
    # json.dump(statistic, open('statistic.json', mode='w'))

if __name__ == '__main__':
    test()