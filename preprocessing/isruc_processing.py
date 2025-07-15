import mne
import numpy as np
import os
import pandas as pd
from edf import read_raw_edf


def read_path():
    data_path = []

    path_ = f"/data/datasets2/ISRUC_extracted/group1/"
    path_list = os.listdir(path_)
    for path in path_list:
        data_path.append(path_ + f"{path}/" + f"{path}")
    print(data_path)
    return list(data_path)


def create_epoch(time, description):
    """
    :param raw_data:
    :param description:
    :return: epoch
    """
    n_event = label.shape[0]
    duration = np.repeat(30, n_event)

    oorig_time = time[0]
    onset = np.arange(oorig_time, n_event*30, 30)
    annotations = mne.Annotations(onset, duration, description)
    raw.set_annotations(annotations)

    events_train, event_id = mne.events_from_annotations(
        raw, chunk_duration=30.)

    # if '?' in event_id.keys():
    #     event_id.pop('?')
    # if 'M' in event_id.keys():
    #     event_id.pop('M')
    # print(event_id)

    tmax = 30. - 1. / raw.info['sfreq']  # tmax in included

    epochs = mne.Epochs(raw=raw, events=events_train, event_id=event_id, tmin=0., tmax=tmax, baseline=None,
                        preload=True)

    epochs.resample(sfreq=100)
    return epochs


if __name__ == '__main__':
    """
    'X1': EMG CHIN
    'X3': Leg1 EMG
    'X4': Leg2 EMG 
    """
    data_path = read_path()
    data_path.remove('/data/datasets2/ISRUC_extracted/group1/40/40')
    data_path.remove('/data/datasets2/ISRUC_extracted/group1/8/8')
    print("dataset length:", len(data_path))
    seq_length = 20

    for i in range(len(data_path)):
        raw = read_raw_edf(data_path[i] + ".rec")
        print(raw.info["ch_names"])

        folder = f""  # save path
        if not os.path.exists(folder):
            os.makedirs(folder)

        _, time = raw[:]

        label = pd.read_table(data_path[i]+"_1.txt", names=["stage"])
        label.replace({"stage": {5: 4}}, inplace=True)
        epochs = create_epoch(time, list(label["stage"]))

        """EEG + EMG"""
        # if "E1-M2" in raw.info["ch_names"]:
        #     if "X1" in raw.info["ch_names"]:
        #         epochs = epochs.pick_channels(['X1', 'X3', 'X4', 'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'])
        #     else:
        #         epochs = epochs.pick_channels(['24', '26', '27', 'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'])
        # else:
        #     if "X1" in raw.info["ch_names"]:
        #         epochs = epochs.pick_channels(['X1', 'X3', 'X4', 'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'])
        #     else:
        #         epochs = epochs.pick_channels(['24', '26', '27', 'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'])


        """EEG + EOG"""
        if "E1-M2" in raw.info["ch_names"]:
            epochs = epochs.pick_channels(['E1-M2', 'E2-M1', 'F3-M2', 'C3-M2', 'O1-M2', 'F4-M1', 'C4-M1', 'O2-M1'])
        else:
            epochs = epochs.pick_channels(['LOC-A2', 'ROC-A1', 'F3-A2', 'C3-A2', 'O1-A2', 'F4-A1', 'C4-A1', 'O2-A1'])


        raw_data = epochs.get_data()
        raw_label = epochs.get_annotations_per_epoch()
        label = []
        for annotation in raw_label:
            anno = annotation[0][2]
            label.append(anno)
        label = np.array(label)
        seq_num = raw_data.shape[0] // seq_length

        if not os.path.exists(folder + f"/{os.path.split(data_path[i])[1]}" + "/data"):
            os.makedirs(folder + f"/{os.path.split(data_path[i])[1]}" + "/data")
        if not os.path.exists(folder + f"/{os.path.split(data_path[i])[1]}" + "/label"):
            os.makedirs(folder + f"/{os.path.split(data_path[i])[1]}" + "/label")

        for j in range(seq_num-1):
            save_data_path = folder + f"/{os.path.split(data_path[i])[1]}" + f"/data/{j}"
            save_label_path = folder + f"/{os.path.split(data_path[i])[1]}" + f"/label/{j}"
            save_data = raw_data[j*seq_length: j*seq_length + seq_length, :, :]
            save_label = label[j*seq_length: j*seq_length + seq_length]
            np.save(save_data_path, save_data)
            np.save(save_label_path, save_label)
    print("Done!!!")
