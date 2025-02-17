import torch.nn as nn
import math
import torch
import copy
import torch.nn.functional as F
from utils.util import fix_randomness
import numpy as np


def clones(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_head):
        super(MultiHeadAttention, self).__init__()
        self.w_query = nn.Linear(input_size, input_size)
        self.w_key = nn.Linear(input_size, input_size)
        self.w_value = nn.Linear(input_size, input_size)
        self.num_head = num_head
        self.dense = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.input_size = input_size
        self.att_dropout = nn.Dropout(0.25)

    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        query = self.w_query(input_tensor)
        key = self.w_key(input_tensor)
        value = self.w_value(input_tensor)
        query = query.view(batch_size, -1, self.num_head, self.input_size//self.num_head).permute(0, 2, 1, 3)
        key = key.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        value = value.view(batch_size, -1, self.num_head, self.input_size // self.num_head).permute(0, 2, 1, 3)
        attention_score = torch.matmul(query, key.transpose(-1, -2))
        attention_score = attention_score / math.sqrt(input_tensor.shape[2])
        attention_prob = nn.Softmax(dim=1)(attention_score)
        attention_prob = self.att_dropout(attention_prob)
        context = torch.matmul(attention_prob, value)
        context = context.permute(0, 2, 1, 3).contiguous()
        context = context.view(batch_size, -1, self.num_head*(self.input_size//self.num_head))
        hidden_state = self.dense(context)
        return hidden_state


class FeedForward(torch.nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.drop(self.relu(self.linear1(x)))
        x = self.drop(self.relu(self.linear2(x)))
        return x


class LayerNorm(torch.nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a1 = nn.Parameter(torch.ones(features))
        self.b1 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a1 * (x - mean) / (std + self.eps) + self.b1


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.pos_emb = PositionalEncoding(size)
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x):
        x = self.pos_emb(x.transpose(0, 1)).transpose(0, 1)
        x = self.sublayer[0](x, lambda x: self.self_attn(x))
        return self.sublayer[1](x, self.feed_forward)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]

        return self.dropout(x)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, layer_num, drop_out, n_head):
        super(MultiHeadAttentionBlock, self).__init__()
        self.multi_attention = MultiHeadAttention(d_model, d_model, n_head)
        self.dropout_rate = drop_out
        self.feedforward = FeedForward(d_model, d_model*4, self.dropout_rate)
        self.encoder = EncoderLayer(d_model, self_attn=self.multi_attention, feed_forward=self.feedforward,
                                    dropout=self.dropout_rate)
        self.layer_num = layer_num

    def forward(self, x):
        for _ in range(self.layer_num):
            x = self.encoder(x)
        return x


def D(p, z, version='simplified'):
    if version == 'original':
        z = z.detach()
        p = F.normalize(p, dim=2)
        z = F.normalize(z, dim=2)
        return -(p * z).sum(dim=2).mean()

    elif version == 'simplified':
        return - F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=128, hidden_dim=512, out_dim=128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch * seq_len, -1)
        x = self.layer1(x)
        x = x.view(batch, seq_len, -1)
        x = self.layer2(x)
        return x


class projection_MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=512, out_dim=128):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.num_layers = 3

    def set_layers(self, num_layers):
        self.num_layers = num_layers

    def forward(self, x):
        batch, seq_len = x.shape[0], x.shape[1]
        x = x.view(batch*seq_len, -1)
        if self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        elif self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        else:
            raise Exception
        x = x.view(batch, seq_len, -1)
        return x


def get_item(rand, signal):
    if rand == 1:
        return flipping(signal)
    elif rand == 2:
        return scaling(signal)
    else:
        return negate(signal)


def augmentation(eeg, eog, args):
    rand = np.random.randint(args["rand"])
    fix_randomness(rand)
    rand1 = np.random.randint(1, 4)
    rand2 = np.random.randint(1, 4)
    while rand1 == rand2:
        rand2 = np.random.randint(1, 4)

    eeg_aug1 = get_item(rand1, eeg)
    eog_aug1 = get_item(rand1, eog)

    eeg_aug2 = get_item(rand2, eeg)
    eog_aug2 = get_item(rand2, eog)

    return eeg_aug1, eog_aug1, eeg_aug2, eog_aug2


def negate(signal):
    negated_signal = signal * (-1)
    return negated_signal


def scaling(x):
    sigma = np.random.uniform(1.1)
    return x*sigma


def flipping(x):
    return torch.flip(x, dims=[1])


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        losses = 0
        for i in range(zis.shape[0]):
            zis_ = zis[i, :, :]
            zjs_ = zjs[i, :, :]

            representations = torch.cat([zjs_, zis_], dim=0)

            similarity_matrix = self.similarity_function(representations, representations)
            # filter out the scores from the positive samples
            self.batch_size = zis_.shape[0]
            l_pos = torch.diag(similarity_matrix, self.batch_size)
            r_pos = torch.diag(similarity_matrix, -self.batch_size)

            positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
            # print(positives.shape)
            # negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)
            negatives = similarity_matrix[self.get_correlated_mask().type(torch.bool)].view(2 * self.batch_size, -1)
            # print(negatives.shape)
            logits = torch.cat((positives, negatives), dim=1)
            logits /= self.temperature

            labels = torch.zeros(2 * self.batch_size).to(self.device).long()
            loss = self.criterion(logits, labels)
            losses += loss
        return losses / (2 * self.batch_size)


class ProjHead(nn.Module):
    def __init__(self, args):
        super(ProjHead, self).__init__()
        self.input_length = 512
        self.classifier = nn.Sequential(
            nn.Linear(self.input_length, self.input_length // 2),
            nn.BatchNorm1d(self.input_length // 2),
            nn.ReLU(True),
            nn.Linear(self.input_length // 2, self.input_length // 4),
        )

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(-1, self.input_length)
        x = self.classifier(x)
        x = x.view(batch, 20, -1)
        return x