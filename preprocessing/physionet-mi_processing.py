import collections

import mne
import numpy as np
import os
import pandas as pd


def read_path():
    data_path = []

    path_ = f"/data/datasets/eeg-motor-movementimagery-dataset-1.0.0/files/"
    path_list = [f"S{str(i).zfill(3)}" for i in range(1, 110)]
    for path in path_list:
        data_path.append(path_ + f"{path}/")
    print(data_path)
    return list(data_path)


if __name__ == '__main__':
    data_path = read_path()
    seconds = []
    consistent = []
    for idx, subject in enumerate(data_path):
        if not os.path.exists(f"/data/datasets2/BCI2000-4/{idx}"):
            os.makedirs(f"/data/datasets2/BCI2000-4/{idx}/data")
            os.makedirs(f"/data/datasets2/BCI2000-4/{idx}/label")
        subject_path_edf = []
        subject_path_list = os.listdir(subject)
        for path in sorted(subject_path_list):
            if path[-3:] == 'edf' and path[-6:-4] in ['04', '06', '08', '10', '12', '14']:
                subject_path_edf.append([subject + f"{path}", subject + f"{path}.event"])
        i = 0
        for file in subject_path_edf:
            edf_file, event_file = file[0], file[1]
            raw = mne.io.read_raw_edf(edf_file)
            print(raw.info['sfreq'])
            event_from_anno, event_dict = mne.events_from_annotations(raw)
            raw_data = raw.get_data()
            event_data_point = event_from_anno[:, 0]
            event_data_type = event_from_anno[:, 2]
            print(event_data_point.shape)
            print(event_data_type.shape)
            print(event_data_type)
            for num in range(event_data_point.shape[0]):
                if event_data_type[num] == 2 or event_data_type[num] == 3:
                    start = event_data_point[num]
                    end = start + 640
                    if end <= raw_data.shape[1]:
                        save_data = raw_data[:, start:end]
                        if event_data_type[num] == 2:
                            if file[0][-6:-4] in ['04', '08', '12']:
                                save_label = 0  # left
                            else:
                                save_label = 2  # fist
                        else:
                            if file[0][-6:-4] in ['04', '08', '12']:
                                save_label = 1  # right
                            else:
                                save_label = 3  # feet
                        np.save(f"/data/datasets2/BCI2000-4/{idx}/data/{i}", save_data)
                        np.save(f"/data/datasets2/BCI2000-4/{idx}/label/{i}", save_label)
                        i += 1
            consistent.append(event_from_anno[1, 0])
            seconds.append(raw_data.shape[1] // raw.info['sfreq'])

    print(collections.Counter(seconds))
    print(collections.Counter(consistent))




