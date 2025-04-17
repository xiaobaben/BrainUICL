import torch
import torch.nn as nn
from utils.config import ModelConfig
from utils.util_block import MultiHeadAttentionBlock
import numpy as np
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class CPC(object):
    def __init__(self, blocks, args):
        super(CPC, self).__init__()
        self.args = args
        self.feature_extractor = blocks[0]
        self.feature_encoder = blocks[1]
        self.classifier = blocks[2]
        self.model_param = ModelConfig(args.dataset)

        self.num_channels = self.model_param.EncoderParam.d_model
        self.d_model = self.model_param.EncoderParam.d_model

        self.timestep = 3
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.Wk = nn.ModuleList([nn.Sequential(nn.Linear(self.d_model, self.d_model*4),
                                               nn.Dropout(0.1),
                                               nn.GELU(),
                                               nn.Linear(self.d_model*4, self.d_model)).to(self.device)
                                 for _ in range(self.timestep)])

        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = MultiHeadAttentionBlock(self.d_model,
                                               self.model_param.EncoderParam.layer_num,
                                               self.model_param.EncoderParam.drop,
                                               self.model_param.EncoderParam.n_head).to(self.device)

        self.optimizer = torch.optim.Adam([
                                           {"params": list(self.feature_extractor.parameters())},
                                           {"params": list(self.feature_encoder.parameters())},
                                           {"params": list(self.encoder.parameters()), "lr": self.args.lr},
                                           {"params": list(self.Wk.parameters()), "lr": self.args.lr}],
                                          lr=self.args.ssl_lr, betas=(self.args.beta1, self.args.beta2),
                                          weight_decay=self.args.weight_decay)

    def update(self, eeg, eog, label):
        # ====== Data =====================
        seq_len = self.model_param.SeqLength
        batch = eeg.shape[0]

        # Src original features
        epoch_size = self.model_param.EpochLength

        eog = eog.view(-1, self.model_param.EogNum, epoch_size)
        eeg = eeg.view(-1, self.model_param.EegNum, epoch_size)

        # EEG + EOG
        eeg_eog_feature = self.feature_extractor(eeg, eog)
        eeg_eog_feature = self.feature_encoder(eeg_eog_feature)  # batch, 20, 128
        t_samples = torch.randint(low=10, high=(seq_len - self.timestep), size=(1,)).long().to(self.device)
        loss = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = eeg_eog_feature[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = eeg_eog_feature[:, :t_samples + 1, :]

        output = self.encoder(forward_seq)  # batch, 15, 128

        c_t = output[:, t_samples, :].view(batch, -1)  # batch, 128

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 5, batch, 128
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # batch, 128
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # batch, 128   128, batch
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1. * batch * self.timestep

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier)


class CPC_BCI2000(object):
    def __init__(self, blocks, args):
        super(CPC_BCI2000, self).__init__()
        self.args = args
        self.feature_extractor = blocks[0]
        self.feature_encoder = blocks[1]
        self.classifier = blocks[2]
        self.model_param = ModelConfig(args.dataset)

        self.num_channels = self.model_param.EncoderParam.d_model
        self.d_model = self.model_param.EncoderParam.d_model

        self.timestep = 3
        self.device = args.device
        self.Wk = nn.ModuleList([nn.Sequential(nn.Linear(self.d_model, self.d_model*4),
                                               nn.Dropout(0.1),
                                               nn.GELU(),
                                               nn.Linear(self.d_model*4, self.d_model)).to(self.device)
                                 for _ in range(self.timestep)])

        self.lsoftmax = nn.LogSoftmax(dim=1)
        self.encoder = MultiHeadAttentionBlock(self.d_model,
                                               self.model_param.EncoderParam.layer_num,
                                               self.model_param.EncoderParam.drop,
                                               self.model_param.EncoderParam.n_head).to(self.device)

        self.optimizer = torch.optim.Adam([
                                           {"params": list(self.feature_extractor.parameters())},
                                           {"params": list(self.feature_encoder.parameters())},
                                           {"params": list(self.encoder.parameters()), "lr": self.args.lr},
                                           {"params": list(self.Wk.parameters()), "lr": self.args.lr}],
                                          lr=self.args.contrastive_lr, betas=(self.args.beta[0],
                                                                                 self.args.beta[1]),
                                          weight_decay=self.args.weight_decay)

    def update(self, x):
        # ====== Data =====================
        seq_len = 10
        batch = x.shape[0]


        # EEG + EOG
        ff = self.feature_extractor(x)
        ff = self.feature_encoder(ff)
        t_samples = torch.randint(low=5, high=(seq_len - self.timestep), size=(1,)).long().to(self.device)
        loss = 0
        encode_samples = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.timestep + 1):
            encode_samples[i - 1] = ff[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = ff[:, :t_samples + 1, :]

        output = self.encoder(forward_seq)  # batch, 15, 128

        c_t = output[:, t_samples, :].view(batch, -1)  # batch, 128  上下文特征

        pred = torch.empty((self.timestep, batch, self.num_channels)).float().to(self.device)  # 5, batch, 128
        for i in np.arange(0, self.timestep):
            linear = self.Wk[i]
            pred[i] = linear(c_t)  # batch, 128
        for i in np.arange(0, self.timestep):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))  # batch, 128   128, batch
            loss += torch.sum(torch.diag(self.lsoftmax(total)))
        loss /= -1. * batch * self.timestep

        self.optimizer.zero_grad()
        loss.backward()

        self.optimizer.step()

        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier)


class BufferPseudoLabelFinetune(object):
    def __init__(self, blocks, teacher_blocks, args):
        super(BufferPseudoLabelFinetune, self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
        self.feature_extractor = blocks[0].to(self.device)
        self.feature_encoder = blocks[1].to(self.device)
        self.classifier = blocks[2].to(self.device)

        self.feature_extractor_t = teacher_blocks[0].to(self.device)
        self.feature_encoder_t = teacher_blocks[1].to(self.device)
        self.classifier_t = teacher_blocks[2].to(self.device)

        self.model_param = ModelConfig(self.args.dataset)

        self.softmax = nn.Softmax(dim=1)
        self.confidence_level = 0.9
        self.optimizer = torch.optim.Adam([{"params": list(self.feature_extractor.parameters())},
                                           {"params": list(self.feature_encoder.parameters())},
                                           {"params": list(self.classifier.parameters())}],
                                          lr=self.args.cl_lr, betas=(self.args.beta1, self.args.beta2),
                                          weight_decay=self.args.weight_decay)

        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, eeg, eog, label):
        # ====== Data =====================
        batch = eeg.shape[0]
        epoch_size = self.model_param.EpochLength

        eog_new = eog[:, :self.model_param.SeqLength, :, :]
        eog_train = eog[:, self.model_param.SeqLength:, :, :]

        eeg_new = eeg[:, :self.model_param.SeqLength, :, :]
        eeg_train = eeg[:, self.model_param.SeqLength:, :, :]

        eeg = torch.concat((eeg_new, eeg_train), dim=0)
        eog = torch.concat((eog_new, eog_train), dim=0)

        eog = eog.view(-1, self.model_param.EogNum, epoch_size)
        eeg = eeg.view(-1, self.model_param.EegNum, epoch_size)

        label_new = label[:, :self.model_param.SeqLength]
        label_train = label[:, self.model_param.SeqLength:]

        eeg_eog_feature = self.feature_extractor(eeg, eog)  # batch, 20, 512
        eeg_eog_feature = self.feature_encoder(eeg_eog_feature)

        eeg_eog_feature_train = eeg_eog_feature[batch:, :, :]
        eeg_eog_feature_new = eeg_eog_feature[:batch, :, :]

        pred_train = self.classifier(eeg_eog_feature_train)
        self.optimizer.zero_grad()

        with torch.no_grad():
            eog_new = eog_new.contiguous().view(-1, self.model_param.EogNum, epoch_size)
            eeg_new = eeg_new.contiguous().view(-1, self.model_param.EegNum, epoch_size)
            ff = self.feature_extractor_t(eeg_new, eog_new)
            ff = self.feature_encoder_t(ff)
            mean_t_pred = self.classifier_t(ff)
            mean_t_pred = mean_t_pred.permute(0, 2, 1)
            mean_t_pred = mean_t_pred.view(-1, 5)
            mean_t_pred = self.softmax(mean_t_pred)  # 640, 5
            pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
            target_pseudo_labels = mean_t_pred.max(1, keepdim=True)[1].squeeze()

        pred_target = self.classifier(eeg_eog_feature_new)
        pred_target = pred_target.permute(0, 2, 1)
        pred_target = pred_target.view(-1, 5)

        confident_pred = pred_target[pred_prob > self.confidence_level]
        confident_labels = target_pseudo_labels[pred_prob > self.confidence_level]
        loss_new = self.cross_entropy(confident_pred, confident_labels.long())
        loss_old = self.cross_entropy(pred_train, label_train.long())

        loss = self.args.alpha*loss_new + (1-self.args.alpha)*loss_old

        loss.backward()
        self.optimizer.step()

        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier), eeg_eog_feature_train

class BufferPseudoLabelFinetune4_BCI2000(object):
    def __init__(self, blocks, teacher_blocks, args):
        super(BufferPseudoLabelFinetune4_BCI2000, self).__init__()

        self.args = args
        self.device = self.arg.device
        self.feature_extractor = blocks[0].to(self.device)
        self.feature_encoder = blocks[1].to(self.device)
        self.classifier = blocks[2].to(self.device)

        self.feature_extractor_t = teacher_blocks[0].to(self.device)
        self.feature_encoder_t = teacher_blocks[1].to(self.device)
        self.classifier_t = teacher_blocks[2].to(self.device)

        self.model_param = ModelConfig(self.args.dataset)

        self.softmax = nn.Softmax(dim=1)
        self.confidence_level = 0.9
        self.optimizer = torch.optim.Adam([{"params": list(self.feature_extractor.parameters())},
                                           {"params": list(self.feature_encoder.parameters())},
                                           {"params": list(self.classifier.parameters())}],
                                          lr=self.args.incremental_lr, betas=(self.args.beta[0], self.args.beta[1]),
                                          weight_decay=self.args.weight_decay)

        self.cross_entropy = nn.CrossEntropyLoss()
    def update(self, x, label):
        # ====== Data =====================
        batch = x.shape[0]
        epoch_size = self.model_param.EpochLength

        x_new = x[:, :self.model_param.BCICn, :]
        x_train = x[:, self.model_param.BCICn:, :]
        # print(label.shape)
        label_new = label[:, :1]
        label_train = label[:, 1:]

        x = torch.concat((x_new, x_train), dim=0)

        eeg_eog_feature = self.feature_extractor(x)  
        eeg_eog_feature = self.feature_encoder(eeg_eog_feature)

        eeg_eog_feature_train = eeg_eog_feature[batch:, :, :]
        eeg_eog_feature_new = eeg_eog_feature[:batch, :, :]

        pred_train = self.classifier(eeg_eog_feature_train)
        self.optimizer.zero_grad()

        with torch.no_grad():
            ff = self.feature_extractor_t(x_new)
            ff = self.feature_encoder_t(ff)
            mean_t_pred = self.classifier_t(ff)
            mean_t_pred = mean_t_pred.view(-1, 4)
 
            mean_t_pred = self.softmax(mean_t_pred)  # 640, 5
            pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
            target_pseudo_labels = mean_t_pred.max(1, keepdim=True)[1].squeeze()

        pred_target = self.classifier(eeg_eog_feature_new)

        pred_target = pred_target.view(-1, 4)

        confident_pred = pred_target[pred_prob > self.confidence_level]
        confident_labels = target_pseudo_labels[pred_prob > self.confidence_level]
        loss_new = self.cross_entropy(confident_pred, confident_labels.long())
        label_train = label_train.view(-1)
        loss_old = self.cross_entropy(pred_train, label_train.long())

        loss = self.args.alpha*loss_new + (1-self.args.alpha)*loss_old
        loss.backward()
        self.optimizer.step()


        return loss.item(), (self.feature_extractor, self.feature_encoder, self.classifier), eeg_eog_feature_train

