import os
import csv
import sys
import argparse
import data
import math
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import sentencepiece as spm

from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from model import LSTMClassifier, BiLSTMClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../data/toy_data/',
                                                        help='data_directory')
    parser.add_argument('--model_dir', type=str, default=None,
                                                        help='model_dirctory')
    parser.add_argument('--target_file', type=str, default=None,
                        help='must give target file name if do_predict=True')
    parser.add_argument('--hidden_dim', type=int, default=32,
                                            help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=32,
                                            help='size for each minibatch')
    parser.add_argument('--num_train_epochs', type=int, default=5,
                                        help='total number of training epochs')
    parser.add_argument('--embedding_dim', type=int, default=128,
                                        help='embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                                            help='weight_decay rate')
    parser.add_argument('--bidirectional', type=bool, default=False,
                                        help='whether to use BiLSTM or not')
    parser.add_argument('--sp_model', type=str, default=None,
                            help='input filename if use sentencepiece model')
    parser.add_argument('--optimizer', type=str, default='Adam',
                                                            help='optimizer')
    parser.add_argument('--do_train', type=bool, default=False,
                                            help='whether to run training')
    parser.add_argument('--do_eval', type=bool, default=False,
                                    help='whether to run eval on the dev set')
    parser.add_argument('--do_test', type=bool, default=False,
                                help='whether to run the model on the test set')
    parser.add_argument('--do_predict', type=bool, default=False,
                                help='whether to run the model on given data')

    args = parser.parse_args()

    if not args.data_dir[-1] == '/':
        args.data_dir = args.data_dir + '/'
    if args.model_dir:
        if not args.model_dir[-1] == '/':
            args.model_dir = args.model_dir + '/'
    if not (args.do_train or args.do_eval or args.do_test or args.do_predict):
        print('At least one command must be True.')
        print('--do_train\n--do_eval\n--do_test\n--do_predict')
        sys.exit()
    return args

class DataProcessor:
    def get_train_examples(self, data_dir):
        return self.create_examples(
                            self.read_tsv(os.path.join(data_dir, 'train.tsv')))

    def get_dev_examples(self, data_dir):
        return self.create_examples(
                            self.read_tsv(os.path.join(data_dir, 'dev.tsv')))

    def get_test_examples(self, data_dir):
        return self.create_examples(
                            self.read_tsv(os.path.join(data_dir, 'test.tsv')))

    def get_target_examples(self, filename):
        words = self.read_tsv(filename)
        examples = defaultdict(list)
        for word in words:
            for w in word:
                examples['chemical'].append(w)
        return examples

    def read_tsv(self, input_file):
        with open(input_file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            lines = []
            for line in reader:
                lines.append(line)

            return lines

    def create_examples(self, lines):
        examples = defaultdict(list)
        for line in lines:
            examples[line[1]].append(line[0])

        return examples


def train(model, model_file, optimizer, train_data, dev_data, batch_size,
                                                            max_epoch, device):
    #train mode
    model = model.train()
    criterion = nn.NLLLoss()
    for epoch in range(max_epoch):
        model = model.train()
        print('Epoch:', epoch+1)
        total_loss = 0
        for batch, labels, lengths in train_data:
            # sorting sequences is necessary for ONNX export
            # convert tensor to cuda if gpu is available
            batch, labels, lengths = data.sort_batch(batch, labels,
                                                            lengths, device)
            model.zero_grad()
            pred = model(batch, lengths)
            batch_loss = criterion(pred, labels)

            batch_loss.backward()
            optimizer.step()

            # torch.max(input, axis): return(tensor(max, max_indices))
            pred_idx = torch.max(pred, 1)

            total_loss += batch_loss.item()

        evaluate_dev_set(model, dev_data, device)

    torch.save(model.state_dict(), model_file)
    print('Training finished.')

    return model


def evaluate_dev_set(model, dev_data, device):
    # eval mode
    model = model.eval()
    criterion = nn.NLLLoss()
    correct = 0
    total_num = 0
    total_loss = 0
    with torch.no_grad():
        for batch, labels, lengths in dev_data:
            batch, labels, lengths = data.sort_batch(batch, labels, lengths, device)

            pred = model(batch, lengths)
            batch_loss = criterion(pred, labels)

            pred_idx = torch.max(pred, 1)[1]
            correct += (pred_idx == labels).sum().item()
            total_num += labels.size(0)

            total_loss += batch_loss.item()

    acc = correct / total_num
    print('Eval loss:', total_loss/total_num, '\t', 'acc:', acc)


def save_results(out_file, lp, words, pred, confidence, true=None):
    results = []
    if true:
        for i in range(len(words)):
            # [word, true, pred]
            results.append([words[i], lp.id2label[int(true[i])],
                            lp.id2label[int(pred[i])], str(confidence[i])[:5]])
    else:
        for i in range(len(words)):
            # [word, true, pred]
            results.append([words[i], lp.id2label[int(pred[i])],
                                                        str(confidence[i])[:5]])

    with open(out_file, 'w') as f:
        for line in results:
            f.write('\t'.join(line)+ '\n')

def evaluate_test_set(model, lp, out_file, test_data, device):
    model = model.eval()
    y_true = []
    y_pred = []
    confidence = []
    words = []
    with torch.no_grad():
        for batch, labels, lengths, raw_data in test_data:
            batch, labels, lengths, raw_data = data.sort_batch(batch, labels,
                                                        lengths, device, raw_data)

            pred = model(batch, lengths)
            pred_idx = torch.max(pred, 1)

            y_true += labels.int().cpu().tolist()
            y_pred += pred_idx[1].int().cpu().tolist()
            # pred_idx[0]: max outputs with LogSoftmax
            # prob(confidence) = e ** LogSoftmax
            confidence += list(math.e**pred_idx[0].detach().cpu().numpy())
            words.extend(raw_data)

    labels=[v for k, v in lp.id2label.items()]
    print('num_test_samples:', len(y_true))
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))
    print(confusion_matrix(y_true, y_pred))
    save_results(out_file, lp, words, y_pred, confidence, y_true)


def predict(model, lp, out_file, target_data, device):
    model = model.eval()
    y_pred = []
    words = []
    confidence = []
    with torch.no_grad():
        for batch, labels, lengths, raw_data in target_data:
            batch, labels, lengths, raw_data = data.sort_batch(batch, labels,
                                                        lengths, device, raw_data)

            pred = model(batch, lengths)
            pred_idx = torch.max(pred, 1)

            y_true += labels.int().cpu().tolist()
            y_pred += pred_idx[1].int().cpu().tolist()
            # pred_idx[0]: max outputs with LogSoftmax
            # prob(confidence) = e ** LogSoftmax
            confidence += list(math.e**pred_idx[0].detach().cpu().numpy())
            words.extend(raw_data)

    labels=[v for k, v in lp.id2label.items()]
    save_results(out_file, lp, words, y_pred, confidence)


def display_samples(dataset, tk, lp):
    print('samples')
    for data in dataset:
        # data[0].size() = (batch_size, max_seq_len)
        # data[1]: labels
        i = 0
        for seq in data[0]:
            ids = seq.cpu().tolist()
            word = tk.DecodeIds(ids)
            print('sample', i+1, '   word:', word,
                        '   label:', lp.id2label[int(data[1][i])])
            print('tokens:', tk.EncodeAsPieces(word))
            print('ids:', ids)
            print('')
            i += 1
            if i == 3:
                break
        break
    print('')


def main():
    args = parse_arg()
    processor = DataProcessor()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('device:', torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print('device: cpu')
    print('')

    if args.bidirectional == True:
        Classifier = BiLSTMClassifier
    else:
        Classifier = LSTMClassifier

    if args.optimizer == 'SGD':
        Optimizer = optim.SGD
    elif args.optimizer == 'Adam':
        Optimizer = optim.Adam
    else:
        print('Optimizer:', args.optimizer, 'is not supported.' )
        sys.exit()

    if args.do_train:
        # get train examples (dict): dict[category] = list(words)
        train_examples = processor.get_train_examples(args.data_dir)
        dev_examples = processor.get_dev_examples(args.data_dir)

        num_train_examples = sum([len(v) for k, v in train_examples.items()])
        num_dev_examples = sum([len(v) for k, v in dev_examples.items()])
        print('num_train_samples:', num_train_examples)
        print('num_eval_samples:', num_dev_examples)

        lp = data.LabelProcessor(train_examples)
        lp.save(args.model_dir)
        labels = lp.label2id

        # tk: Tokenizer
        if args.sp_model:
            tk = spm.SentencePieceProcessor()
            tk.load(args.sp_model)

        # build char vocabulary if sp-model is not given
        if not args.sp_model:
            cv = data.CharVocabBuilder(train_examples)
            cv.save_vocab(args.model_dir)
            tk = data.CharTokenizer(args.model_dir)

        vocab_size = tk.GetPieceSize()
        model_file = args.model_dir + 'lstm.model'
        # train_data(DataLoader)
        train_data = data.create_batched_tensor(tk, lp, train_examples,
                                    batch_size=args.batch_size, raw_text=False)
        dev_data = data.create_batched_tensor(tk, lp, dev_examples,
                                    batch_size=args.batch_size, raw_text=False)

        display_samples(train_data, tk, lp)

        model = Classifier(vocab_size, args.embedding_dim,
                                args.hidden_dim, len(labels),  args.batch_size)
        model.to(device)
        #optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
        optimizer = Optimizer(model.parameters(), lr=args.learning_rate,
                                                weight_decay=args.weight_decay)

        model = train(model, model_file, optimizer, train_data, dev_data,
                                args.batch_size, args.num_train_epochs, device)

    if args.do_eval:
        # get_dev_examples(dict): dict[category] = list(words)
        dev_examples = processor.get_dev_examples(args.data_dir)

        num_dev_examples = sum([len(v) for k, v in dev_examples.items()])
        print('num_eval_samples:', num_dev_examples)

        if args.do_train == False:
            lp = data.LabelProcessor()
            lp.load(args.model_dir)
            labels = lp.label2id
            # tk: Tokenizer
            if args.sp_model:
                tk = spm.SentencePieceProcessor()
                tk.load(args.sp_model)

            # load character tokenizer
            if not args.sp_model:
                tk = data.CharTokenizer(args.model_dir)

            vocab_size = tk.GetPieceSize()
            try:
                model = Classifier(vocab_size, args.embedding_dim,
                                args.hidden_dim, len(labels), args.batch_size)
            except:
                model = BiLSTMClassifier(vocab_size, args.embedding_dim,
                                args.hidden_dim, len(labels), args.batch_size)
            model_file = args.model_dir + 'lstm.model'
            model.load_state_dict(torch.load(model_file,
                                            map_location=torch.device(device)))
            model.to(device)

        dev_data = data.create_batched_tensor(tk, lp, dev_examples,
                                    batch_size=args.batch_size, raw_text=False)
        display_samples(dev_data, tk, lp)

        evaluate_dev_set(model, dev_data, device)

    if args.do_test:
        test_examples = processor.get_test_examples(args.data_dir)
        num_test_examples = sum([len(v) for k, v in test_examples.items()])
        print('num_test_samples:', num_test_examples)

        # out_file(result file) = '[data_dir]/[model_name]_result.tsv'
        out_file = args.data_dir + args.model_dir.split('/')[-2] + '_result.tsv'

        if args.do_train == False:
            lp = data.LabelProcessor()
            lp.load(args.model_dir)
            labels = lp.label2id
            # tk: Tokenizer
            if args.sp_model:
                tk = spm.SentencePieceProcessor()
                tk.load(args.sp_model)

            # load character tokenizer
            if not args.sp_model:
                tk = data.CharTokenizer(args.model_dir)

            vocab_size = tk.GetPieceSize()
            try:
                model = Classifier(vocab_size, args.embedding_dim,
                                args.hidden_dim, len(labels), args.batch_size)
            except:
                model = BiLSTMClassifier(vocab_size, args.embedding_dim,
                                args.hidden_dim, len(labels), args.batch_size)
            model_file = args.model_dir + 'lstm.model'
            model.load_state_dict(torch.load(model_file,
                                            map_location=torch.device(device)))
            model.to(device)

        test_data = data.create_batched_tensor(tk, lp, test_examples,
                                                batch_size=args.batch_size)
        display_samples(test_data, tk, lp)

        evaluate_test_set(model, lp, out_file, test_data, device)

    if args.do_predict:
        pred_examples = processor.get_target_examples(args.target_file)
        num_pred_examples = sum([len(v) for k, v in pred_examples.items()])
        print('num_target_samples:', num_pred_examples)

        model_name = '/' + args.model_dir.split('/')[-2]
        path_name = '/'.join(args.target_file.split('/')[:-1]) + model_name
        out_file = path_name + '_result.tsv'
        os.makedirs(path_name, exist_ok=True)

        lp = data.LabelProcessor()
        lp.load(args.model_dir)
        labels = lp.label2id
        # tk: Tokenizer
        if args.sp_model:
            tk = spm.SentencePieceProcessor()
            tk.load(args.sp_model)

        # load character tokenizer
        if not args.sp_model:
            tk = data.CharTokenizer(args.model_dir)

        vocab_size = tk.GetPieceSize()
        try:
            model = Classifier(vocab_size, args.embedding_dim,
                            args.hidden_dim, len(labels), args.batch_size)
        except:
            model = BiLSTMClassifier(vocab_size, args.embedding_dim,
                            args.hidden_dim, len(labels), args.batch_size)
        model_file = args.model_dir + 'lstm.model'
        model.load_state_dict(torch.load(model_file,
                                        map_location=torch.device(device)))
        model.to(device)

        target_data = data.create_batched_tensor(tk, lp, pred_examples,
                                                batch_size=args.batch_size)
        display_samples(target_data, tk, lp)
        #evaluate_test_set(model, lp, out_file, test_data)
        predict(model, lp, out_file, target_data, device)



if __name__=='__main__':
    main()
