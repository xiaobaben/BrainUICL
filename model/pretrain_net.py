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
