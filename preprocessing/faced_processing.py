import pandas as pd
import os
import numpy as np


save_path = "/data/datasets2/Face/"
ori_path = f"/data/cyn/FACED/Processed_data"

path = [str(i) for i in range(0, 123)]
path = [i.zfill(3) for i in path]
for idx in path:
    file_path = ori_path + f"/sub{idx}.pkl"
    x_data = pd.read_pickle(file_path)

    y_data = []
    for y in range(9):
        if y == 4:
            y_data.extend([y for _ in range(4)])
        else:
            y_data.extend([y for _ in range(3)])
    for i in range(28):
        if not os.path.exists(save_path + f"{int(idx)}/data"):
            os.makedirs(save_path + f"{int(idx)}/data")
        if not os.path.exists(save_path + f"{int(idx)}/label"):
            os.makedirs(save_path + f"{int(idx)}/label")
        save_x_path = save_path + f"{int(idx)}/data/{i}"
        save_y_path = save_path + f"{int(idx)}/label/{i}"
        save_x = x_data[i, :, :]
        save_y = np.array(y_data[i])
        np.save(save_x_path, save_x)
        np.save(save_y_path, save_y)
    print(f"sub{idx} Done")
