import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size,
                    batch_size, num_layers=1, dropout=0.1, reset_state=False):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                                            batch_first=True)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, batch, lengths):
        # embeds.size() = (batch_size, len(seq), embedding_dim)
        embeds = self.embedding(batch)
        # enforce_sorted = True is only necessary for ONNX export.
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True)
        # hn.size() = (1, batch_size, hidden_dim)
        _, (hn, cn) = self.lstm(packed_input)

        # hn.size() = (batch_size, hidden_dim)
        output = self.dropout_layer(hn[-1])
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size,
                                        batch_size, num_layers=1, dropout=0.2):
        super(BiLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                                        batch_first=True, bidirectional=True)

        self.hidden2out = nn.Linear(hidden_dim*2, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

        self.dropout_layer = nn.Dropout(p=dropout)


    def forward(self, batch, lengths):
        # embeds.size() = (batch_size, len(seq), embedding_dim)
        embeds = self.embedding(batch)
        # enforce_sorted = True is only necessary for ONNX export.
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True)
        # hn.size() = (num_layers * num_directions, batch_size, hidden_dim)
        _, (hn, cn) = self.lstm(packed_input)

        # hn.size() = (num_layers * 2, batch_size, hidden_dim)
        output = self.dropout_layer(torch.cat([hn[-2], hn[-1]], dim=1))
        output = self.hidden2out(output)
        output = self.softmax(output)

        return output
