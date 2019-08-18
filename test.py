import torch
import torch.nn as nn
import torch.utils.data as torch_data
import numpy as np
import pandas as pd
from tqdm import tqdm

from pytorch_transformers import BertModel, load_tf_weights_in_bert, BertConfig, BertForPreTraining, BertTokenizer
from model import convert_tf_checkpoint_to_pytorch, Model, Dataset

ID2CATEGORY = [
    '价格',
    '使用体验',
    '其他',
    '功效',
    '包装',
    '尺寸',
    '成分',
    '整体',
    '新鲜度',
    '服务',
    '气味',
    '物流',
    '真伪'
]
ID2POLARITY = [
    '正面',
    '中性',
    '负面'
]

def load_review_dataset(review_filepath):
    review_df = pd.read_csv(review_filepath, index_col='id')
    dataset = dict()
    for idx, row in review_df.iterrows():
        review = row['Reviews']
        dataset[idx] = review
    return dataset

# data only contain review
class test_collate_fn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, batch_data):
        batch_idx = []
        batch_X = []
        len_X = []
        mask = []
        for idx, review in batch_data:
            batch_idx.append(idx)
            x = self.tokenizer.encode('[CLS]' + review)
            batch_X.append(torch.tensor(x))
            len_X.append(len(x))
            mask.append(torch.ones(len(x), dtype=torch.uint8))
        mask = nn.utils.rnn.pad_sequence(mask, batch_first=True, padding_value=0)
        batch_X = nn.utils.rnn.pad_sequence(batch_X, batch_first=True, padding_value=0)
        return batch_X, len_X, mask, batch_idx

def main():
    pred_file_path = 'test.csv'
    load_save_model = True
    lr = 1e-5
    batch_size = 8
    gpu = True
    torch.manual_seed(0)
    device = torch.device('cpu')
    if gpu:
        device = torch.device('cuda')

    dataset = load_review_dataset('TRAIN/TEST/Test_reviews.csv')

    tokenizer = BertTokenizer(vocab_file='publish/vocab.txt', max_len=512)
    dataset = Dataset(list(dataset.items()))
    dataloader = torch_data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=test_collate_fn(tokenizer)
    )
    bert_pretraining = convert_tf_checkpoint_to_pytorch('./publish/bert_model.ckpt', './publish/bert_config.json')
    model = Model(bert_pretraining.bert)

    model = model.cuda()
    if load_save_model:
        model.load_state_dict(torch.load('./save_model/best.model'))

    pred_file = open(pred_file_path, mode='w', encoding='utf-8')

    pbar = tqdm()
    model.eval()
    for step, (batch_X, len_X, mask, batch_idx) in enumerate(dataloader):
        batch_X = batch_X.to(device)
        mask = mask.to(device)

        scores, gather_idx = model(batch_X, len_X, mask, None)
        (pred_seq_target, pred_match_target, pred_single_aspect_category_target, pred_single_opinion_category_target,\
            pred_cross_category_target, pred_single_aspect_polarity_target, pred_single_opinion_polarity_target,\
                pred_cross_polarity_target) = model.infer(scores, mask)

        label = []

        aspect_idx, opinion_idx = gather_idx
        for b in range(batch_X.shape[0]):
            _aspect_idx, _opinion_idx = aspect_idx[b], opinion_idx[b]
            if len(_aspect_idx) == 0 and len(_opinion_idx) == 0:
                label.append((batch_idx[b], '_', '_', '_', '_'))

            _aspect_cross, _opinion_cross = [False for i in range(len(_aspect_idx))], [False for i in range(len(_opinion_idx))]
            for i in range(len(_aspect_idx)):
                for j in range(len(_opinion_idx)):
                    if pred_match_target[b][i, j] == 1:
                        _aspect_cross[i] = True
                        _opinion_cross[j] = True
                        category = ID2CATEGORY[pred_cross_category_target[b][i, j]]
                        polarity = ID2POLARITY[pred_cross_polarity_target[b][i, j]]
                        aspect = tokenizer.decode(list(batch_X[b, _aspect_idx[i]].cpu().detach().numpy())).replace(' ', '')
                        opinion = tokenizer.decode(list(batch_X[b, _opinion_idx[j]].cpu().detach().numpy())).replace(' ', '')
                        aspect_beg = len(tokenizer.decode(list(batch_X[b, 1:_aspect_idx[i][0]].cpu().detach().numpy())).replace(' ', ''))
                        aspect_end = aspect_beg + len(aspect)
                        opinion_beg = len(tokenizer.decode(list(batch_X[b, 1:_opinion_idx[j][0]].cpu().detach().numpy())).replace(' ', ''))
                        opinion_end = opinion_beg + len(opinion)
                        label.append((batch_idx[b], aspect, opinion, category, polarity))
            for i in range(len(_aspect_idx)):
                if _aspect_cross[i] == False:
                    category = ID2CATEGORY[pred_single_aspect_category_target[b][i]]
                    polarity = ID2POLARITY[pred_single_aspect_polarity_target[b][i]]
                    aspect = tokenizer.decode(list(batch_X[b, _aspect_idx[i]].cpu().detach().numpy())).replace(' ', '')
                    aspect_beg = len(tokenizer.decode(list(batch_X[b, 1:_aspect_idx[i][0]].cpu().detach().numpy())).replace(' ', ''))
                    aspect_end = aspect_beg + len(aspect)
                    label.append((batch_idx[b], aspect, '_', category, polarity))
            for i in range(len(_opinion_idx)):
                if _opinion_cross[i] == False:
                    category = ID2CATEGORY[pred_single_opinion_category_target[b][i]]
                    polarity = ID2POLARITY[pred_single_opinion_polarity_target[b][i]]
                    opinion = tokenizer.decode(list(batch_X[b, _opinion_idx[i]].cpu().detach().numpy())).replace(' ', '')
                    opinion_beg = len(tokenizer.decode(list(batch_X[b, 1:_opinion_idx[i][0]].cpu().detach().numpy())).replace(' ', ''))
                    opinion_end = opinion_beg + len(opinion)
                    label.append((batch_idx[b], '_', opinion, category, polarity))

        for _label in label:
            _label = ','.join(list(map(lambda x: str(x), _label)))
            pred_file.write(_label + '\n')
        pbar.update(batch_size)
        pbar.set_description('step: %d' % step)
    pred_file.close()
    pbar.close()

import json
if __name__ == '__main__':
    main()
    # data = json.load(open('statistic.json', mode='r'))
    # print(1)