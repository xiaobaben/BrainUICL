from utils.config import ModelConfig
from utils.util_block import MultiHeadAttentionBlock
import torch
import torch.nn as nn


class FeatureExtractor(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor, self).__init__()
        self.ModelParam = ModelConfig(args.dataset)
        self.FEBlock_EEG = nn.Sequential(
                    nn.Conv1d(self.ModelParam.EegNum, 64, kernel_size=50, stride=6, bias=False),
                    nn.BatchNorm1d(64),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=8, stride=8),
                    nn.Dropout(0.1),

                    nn.Conv1d(64, 128, kernel_size=8),
                    nn.BatchNorm1d(128),
                    nn.GELU(),

                    nn.Conv1d(128, 256, kernel_size=8),
                    nn.BatchNorm1d(256),
                    nn.GELU(),

                    nn.Conv1d(256, 512, kernel_size=8),
                    nn.BatchNorm1d(512),
                    nn.GELU(),
                    nn.MaxPool1d(kernel_size=4, stride=4),
                )

        self.FEBlock_EOG = nn.Sequential(
            nn.Conv1d(self.ModelParam.EogNum, 64, kernel_size=50, stride=6, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=8, stride=8),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=8),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Conv1d(128, 256, kernel_size=8),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Conv1d(256, 512, kernel_size=8),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.fusion = nn.Linear(1024, 512)

    def forward(self, eeg, eog):
        batch = eeg.shape[0] // self.ModelParam.SeqLength
        eeg = self.FEBlock_EEG(eeg)

        eog = self.FEBlock_EOG(eog)

        eeg = self.avg(eeg).view(batch * self.ModelParam.SeqLength, 1, self.ModelParam.EncoderParam.d_model)
        eog = self.avg(eog).view(batch * self.ModelParam.SeqLength, 1, self.ModelParam.EncoderParam.d_model)

        x = self.fusion(torch.concat((eeg, eog), dim=2))

        x = x.view(batch, self.ModelParam.SeqLength, -1)

        return x


class TransformerEncoder(torch.nn.Module):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()
        self.ModelParam = ModelConfig(args.dataset)
        self.encoder = MultiHeadAttentionBlock(self.ModelParam.EncoderParam.d_model,
                                               self.ModelParam.EncoderParam.layer_num,
                                               self.ModelParam.EncoderParam.drop,
                                               self.ModelParam.EncoderParam.n_head)

    def forward(self, x):
        return self.encoder(x)


class SleepMLP(nn.Module):
    def __init__(self, args):
        super(SleepMLP, self).__init__()
        self.ModelParam = ModelConfig(args.dataset)
        self.dropout_rate = self.ModelParam.SleepMlpParam.drop
        self.sleep_stage_mlp = nn.Sequential(
            nn.Linear(self.ModelParam.SleepMlpParam.first_linear[0],
                      self.ModelParam.SleepMlpParam.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(self.ModelParam.SleepMlpParam.second_linear[0],
                      self.ModelParam.SleepMlpParam.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.sleep_stage_classifier = nn.Linear(self.ModelParam.SleepMlpParam.out_linear[0],
                                                self.ModelParam.SleepMlpParam.out_linear[1], bias=False)

    def forward(self, x):
        x = self.sleep_stage_mlp(x)
        x = self.sleep_stage_classifier(x)
        x = x.permute(0, 2, 1)
        return x


class FeatureExtractor_BCI2000(nn.Module):
    def __init__(self, args):
        super(FeatureExtractor_BCI2000, self).__init__()
        self.ModelParam = ModelConfig(args.dataset)
        self.eeg_fe = nn.Sequential(
            nn.Conv1d(self.ModelParam.BCICn, 64, kernel_size=50, stride=6, bias=False, padding=24),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(0.1),

            nn.Conv1d(64, 128, kernel_size=6, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=6, padding=3),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(256, 512, kernel_size=6, padding=3),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),
        )

        self.avg = nn.AdaptiveAvgPool1d(output_size=1)


    def forward(self, x):
        batch = x.shape[0]
        for i in range(10):
            tp = x[:, :, i * 64:i * 64 + 64]
            tp = tp.view(batch, 1, self.ModelParam.BCICn, 64)
            if i == 0:
                sequence = tp
                sequence = sequence
            else:
                sequence = torch.concat((sequence, tp), dim=1)
        # print(sequence.shape)
        sequence = sequence.view(-1, self.ModelParam.BCICn, 64)
        x1 = self.eeg_fe(sequence)
        x1 = self.avg(x1).view(batch, 10, -1)

        return x1


class TransformerEncoder_BCI2000(torch.nn.Module):
    def __init__(self, args):
        super(TransformerEncoder_BCI2000, self).__init__()
        self.ModelParam = ModelConfig(args.dataset.)
        self.args = args
        self.encoder = MultiHeadAttentionBlock(self.ModelParam.EncoderParam.d_model,
                                               self.ModelParam.EncoderParam.layer_num,
                                               self.ModelParam.EncoderParam.drop,
                                               self.ModelParam.EncoderParam.n_head)

    def forward(self, x):
        return self.encoder(x)  # batch, 28, 512


class MIMLP_BCI2000(nn.Module):
    def __init__(self, args):
        super(MIMLP_BCI2000, self).__init__()
        self.ModelParam = ModelConfig(args.dataset)
        self.dropout_rate = 0.1
        self.mi_mlp = nn.Sequential(
            nn.Linear(5120,
                      self.ModelParam.BCI2000MlpParam.first_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
            nn.Linear(self.ModelParam.BCI2000MlpParam.second_linear[0],
                      self.ModelParam.BCI2000MlpParam.second_linear[1]),
            nn.Dropout(self.dropout_rate),
            nn.GELU(),
        )
        self.mi_classifier = nn.Linear(self.ModelParam.BCI2000MlpParam.out_linear[0],
                                                self.ModelParam.BCI2000MlpParam.out_linear[1], bias=False)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(batch, -1)
        x = self.mi_mlp(x)
        x = self.mi_classifier(x)
        x = x.view(-1, 4)
        return x
