from torch.utils.data import Dataset
import numpy as np
import torch


class BuildDataset(Dataset):
    def __init__(self, data_path, dataset):
        self.data_path = data_path
        self.dataset = dataset
        self.len = len(self.data_path[0])

    def __getitem__(self, index):
        x_data = np.load(self.data_path[0][index])
        y_data = np.load(self.data_path[1][index])
        x_data = torch.from_numpy(np.array(x_data).astype(np.float32))
        y_data = torch.from_numpy(np.array(y_data).astype(np.float32))
        eog = x_data[:, :2, :]
        eeg = x_data[:, 2:, :]

        return eog, eeg, y_data

    def __len__(self):
        return self.len


class BuildBufferDataset(Dataset):
    def __init__(self, new_path, train_path, dataset, args):
        self.new_path = new_path
        self.dataset = dataset
        self.len = len(self.new_path[0])

        self.old_len = int(0.8*self.len)
        self.new_len = int(0.2 * self.len)

        if self.new_len < len(train_path[0]) - args.train_len:
            old_sample_idx = list(np.random.choice(range(args.train_len), self.old_len, replace=False))
            new_sample_idx = list(np.random.choice(range(args.train_len, len(train_path[0])),
                                                   self.new_len, replace=False))
            sample_idx = []
            for x in old_sample_idx:
                sample_idx.append(x)
            for y in new_sample_idx:
                sample_idx.append(y)
        else:
            sample_idx = list(np.random.choice(range(len(train_path[0])), self.len, replace=False))
        if len(sample_idx) < self.len:
            another = list(np.random.choice(range(args.train_len), 1, replace=False))
            sample_idx.append(another[0])
        self.train_path_data = [train_path[0][i-1] for i in sample_idx]
        self.train_path_label = [train_path[1][i-1] for i in sample_idx]

    def __getitem__(self, index):
        x_data_new = np.load(self.new_path[0][index])
        y_data_new = np.load(self.new_path[1][index])
        x_data_new = torch.from_numpy(np.array(x_data_new).astype(np.float32))
        y_data_new = torch.from_numpy(np.array(y_data_new).astype(np.float32))
        # print(index, len(self.train_path_data))
        x_data_train = np.load(self.train_path_data[index])
        y_data_train = np.load(self.train_path_label[index])
        x_data_train = torch.from_numpy(np.array(x_data_train).astype(np.float32))
        y_data_train = torch.from_numpy(np.array(y_data_train).astype(np.float32))

        eog_new = x_data_new[:, :2, :]
        eeg_new = x_data_new[:, 2:, :]

        eog_train = x_data_train[:, :2, :]
        eeg_train = x_data_train[:, 2:, :]

        eog = torch.concat((eog_new, eog_train), dim=0)
        eeg = torch.concat((eeg_new, eeg_train), dim=0)
        y_data = torch.concat((y_data_new, y_data_train), dim=0)
        return eog, eeg, y_data

    def __len__(self):
        return self.len


class Builder(object):
    def __init__(self, data_path, args):
        super(Builder, self).__init__()
        self.data_set = args.dataset
        self.data_path = data_path
        self.Dataset = BuildDataset(self.data_path, self.data_set)
        self.BufferDataset = BuildBufferDataset(self.data_path, args.train_path, self.data_set, args)
