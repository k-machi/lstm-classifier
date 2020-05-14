import os
import csv
import sys
import argparse

import torch
import torch.nn.utils.rnn as rnn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


class CharVocabConstructor:

    def __init__(self, train_data):
        # train_data(dict): dict[category] = list(words)

        self.token_set = self.get_token_set(train_data)

        self.token2id = self.set2id(self.token_set, '<pad>', '<unk>')
        self.id2token = {v: k for k, v in self.token2id.items()}
        #self.label2id = self.set2id(set(self.label_set))

    def get_token_set(self, train_data):
        token_set = set()
        for category in train_data:
            for word in train_data[category]:
                for letter in word:
                    token_set.add(letter)
        return token_set

    def set2id(self, item_set, pad=None, unk=None):
        item2id = defaultdict(int)
        if pad is not None:
            item2id[pad] = 0
        if unk is not None:
            item2id[unk] = 1

        for item in sorted(item_set):
            item2id[item] = len(item2id)
        return item2id

    def save_vocab(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        out_file = model_dir + 'vocab.tsv'
        with open(out_file, 'w') as f:
            for id, token in self.id2token.items():
                f.write(str(id) + '\t' + token + '\n')


class CharTokenizer:

    def __init__(self, model_dir):
        # token2id(dict): dict[token] = int(id)
        # id2token(dict): dict[int(id)] = token
        self.token2id, self.id2token = self.load(model_dir)

    def EncodeAsPieces(self, text):
        seq = []
        for char in text:
            seq.append(char)
        return seq

    def EncodeAsIds(self, text):
        ids = []
        i=0
        for char in text:
            if char in self.token2id:
                ids.append(self.token2id[char])
            else:
                ids.append(self.token2id['<unk>'])
            i+=1
        return ids

    def DecodePieces(self, seqs):
        return ''.join(char_seq)

    def DecodeIds(self, ids):
        seq = []
        for id in ids:
            if id == 0: # <pad>
                break
            seq.append(self.id2token[id])
        return ''.join(seq)

    def GetPieceSize(self):
        return len(self.token2id)

    def load(self, model_dir):
        vocab_file = model_dir + 'vocab.tsv'
        id2token = defaultdict()
        with open(vocab_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            id2token = {}
            for line in reader:
                # line = [id, token]
                id2token[int(line[0])] = line[1]
        token2id = {v: k for k, v in id2token.items()}
        return token2id, id2token

class TrainTensorDataset(Dataset):
    # data_tensor(Tensor): sample data
    # target_tensor(Tensor): sample labels
    # langth_ tensor(Tensor): sample seq length
    def __init__(self, data_tensor, label_tensor, length_tensor):

        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.length_tensor = length_tensor

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return (self.data_tensor[idx], self.label_tensor[idx],
                                                    self.length_tensor[idx])


class TensorDataset(Dataset):
    # data_tensor(Tensor): sample data
    # target_tensor(Tensor): sample labels
    # langth_ tensor(Tensor): sample seq length
    # raw_data(list): list(word)
    def __init__(self, data_tensor, label_tensor, length_tensor, raw_data):

        self.data_tensor = data_tensor
        self.label_tensor = label_tensor
        self.length_tensor = length_tensor
        self.raw_data = raw_data

    def __len__(self):
        return self.data_tensor.size(0)

    def __getitem__(self, idx):
        return (self.data_tensor[idx], self.label_tensor[idx],
                                    self.length_tensor[idx], self.raw_data[idx])


class LabelProcessor:

    def __init__(self, train_data=None):
        # train_data(dict): dict[category] = list(words)

        if train_data:
            label_set = set(train_data.keys())
            self.label2id = self.set2id(label_set)
            self.id2label = {v: k for k, v in self.label2id.items()}

        #self.label2id = self.set2id(set(self.label_set))

    def get_token_set(self, train_data):
        token_set = set()
        for category in train_data:
            for word in train_data[category]:
                for letter in word:
                    token_set.add(letter)
        return token_set

    def set2id(self, item_set):
        item2id = defaultdict(int)
        for item in sorted(item_set):
            item2id[item] = len(item2id)
        return item2id

    def save(self, model_dir):
        os.makedirs(model_dir, exist_ok=True)
        out_file = model_dir + 'label.tsv'
        with open(out_file, 'w') as f:
            for id, label in self.id2label.items():
                f.write(str(id) + '\t' + label + '\n')

    def load(self, model_dir):
        vocab_file = model_dir + 'label.tsv'
        self.id2label = defaultdict()
        with open(vocab_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for line in reader:
                # line = [id, token]
                self.id2label[int(line[0])] = line[1]
        self.label2id = {v: k for k, v in self.id2label.items()}
        return


def get_id_tensor(tokenizer, lp, dataset, raw_text=False):
    # dataset(dict): dict[category] = list(words)
    # labelid(dict): dict[label] = int(id)
    data_seqs = []
    label_ids = []
    seq_len = []
    raw_data = []
    labelid = lp.label2id
    for label in dataset:
        for word in dataset[label]:
            seq_ids = tokenizer.EncodeAsIds(word)
            label_id = labelid[label]
            data_seqs.append(seq_ids)
            label_ids.append(label_id)
            seq_len.append(len(seq_ids))
            if len(seq_ids)<=0:
                print('Error: word length < 0')
                print(seq_ids, label_id, word, len(seq_ids))
                sys.exit()
            raw_data.append(word)
    seq_tensor = padded_seq(data_seqs, max(seq_len))
    label_ids = torch.LongTensor(label_ids)
    seq_len_tensor = torch.LongTensor(seq_len)

    if raw_text:
        return seq_tensor, label_ids, seq_len_tensor, raw_data
    else:
        return seq_tensor, label_ids, seq_len_tensor


def padded_seq(data_seqs, max_seq_len):
    padded_tensor = torch.zeros(len(data_seqs), max_seq_len).long()
    for i in range(len(data_seqs)):
        padded_seq = data_seqs[i]
        padded_seq.extend([0]*(max_seq_len-len(data_seqs[i])))
        padded_tensor[i] = torch.LongTensor(padded_seq)

    return padded_tensor

def create_batched_tensor(tokenizer, lp, dataset, batch_size=32,
                                                raw_text=True, shuffle=True):
    # seq_ids(Tensor): (len(data), len(max_seq))
    # label_ids(Tensor): len(data)
    # seq_len(Tensor):
    # raw_data(list):

    # raw_data is not included when training mode
    if raw_text:
        seq_tensor, label_ids, seq_len, raw_data = get_id_tensor(tokenizer,
                                                    lp, dataset, raw_text=True)
        td = TensorDataset(seq_tensor, label_ids, seq_len, raw_data)
    else:
        seq_tensor, label_ids, seq_len = get_id_tensor(tokenizer, lp, dataset)
        td = TrainTensorDataset(seq_tensor, label_ids, seq_len)

    return DataLoader(td, batch_size, shuffle=shuffle)

def sort_batch(batch, labels, lengths, device, raw_data=None):
    seq_lengths, idx = lengths.sort(0, descending=True)
    seq_lengths = seq_lengths.to(device)
    seq_tensor = batch[idx].to(device)
    label_tensor = labels[idx].to(device)
    if raw_data:
        ret_data = []
        for i in idx:
            ret_data.append(raw_data[int(i)])
        return seq_tensor, label_tensor, seq_lengths, ret_data
    else:
        return seq_tensor, label_tensor, seq_lengths
