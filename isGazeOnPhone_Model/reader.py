import numpy as np
import cv2
import os
from torch.utils.data import Dataset, DataLoader
import random
import torch

class loader(Dataset):
    def __init__(self, path, root,header=True):
        self.lines = []
        if isinstance(path, list):
            for i in path:
                with open(i) as f:
                    line = f.readlines()
                    if header: line.pop(0)
                    for k in range(len(line)):
                        # line_list = line[k].split(" ")
                        # print(line_list[6])
                        # if line_list[6] == "Photo":
                        self.lines.append(line[k])

        else:
            with open(path) as f:
                line = f.readlines()
                if header: self.line.pop(0)
                for j in range(len(line)):
                    # line_list = line[j].split(" ")
                    # print(line_list[6])
                    # if line_list[6] == "Photo":
                    self.lines.append(line[j])


        self.root = root

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        line = line.strip().split(" ")

        face_img = line[0]
        label_ = line[1]

        label_ = np.array(label_).astype("int")
        label = torch.from_numpy(label_).type(torch.int)

        fimg = cv2.imread(os.path.join(self.root, face_img))
        fimg = cv2.resize(fimg, (224, 224)) / 255.0
        fimg = fimg.transpose(2, 0, 1)


        img = {
                "face": torch.from_numpy(fimg).type(torch.FloatTensor),
                "label": label,
                }

        return img


def txtload(labelpath, imagepath, batch_size, shuffle=True, num_workers=0, header=True):
    dataset = loader(labelpath, imagepath, header)
    print(f"[Read Data]: Total num: {len(dataset)}")
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load


if __name__ == "__main__":
    image = r"/disk1/repository/isGazeOnPhone/output/image"
    label = r"/disk1/repository/isGazeOnPhone/output/label/test"
    trains = os.listdir(label)
    trains = [os.path.join(label, j) for j in trains]
    # print(trains)
    d = txtload(trains, image, 10)
    print(len(d))
    # (data,label) = d.__iter__()
    for data in d:
        print(data["label"])