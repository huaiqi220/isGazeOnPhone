import model2
import numpy as np
import cv2 
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import yaml
import os
import copy
import math
import reader

# def dis(p1, p2):
#     return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0]) + (p1[1]-p2[1])*(p1[1]-p2[1]))

if __name__ == "__main__":
    config = yaml.safe_load(open("config.yaml"))
    configt = config["train"]
    config = config["test"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["load"]["model_name"]
    load_path = os.path.join(config["load"]["load_path"])
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    #device = torch.device("cpu")
    save_name="evaluation"

    print(f"Test Set: tests")

    tests = os.listdir(labelpath)
    tests = [os.path.join(labelpath, j) for j in tests]

    now_path = "bs_"  + str(configt["params"]["batch_size"]) + "_fix_lr_" + str(str(configt["params"]["lr"]))  + "_ep_" + str(str(configt["params"]["epoch"])) +"_rot_data2"

    save_path = os.path.join(load_path, "checkpoint",now_path)

    if not os.path.exists(os.path.join(load_path, save_name)):
        os.makedirs(os.path.join(load_path, save_name))

    print("Read data")
    dataset = reader.txtload(tests, imagepath, 32, shuffle=True, num_workers=0, header=True)

    begin = config["load"]["begin_step"]
    end = config["load"]["end_step"]
    step = config["load"]["steps"]
    epoch_log = open(os.path.join(load_path, f"{save_name}/epoch.log"), 'a')
    for save_iter in range(begin, end+step, step):
        print("Model building")
        # net = model.ResNet34(2)
        net = model2.Resnet(model2.basic_block,[2,2,2,2],2)
        net = nn.DataParallel(net)
        state_dict = torch.load(os.path.join(save_path, f"Iter_{save_iter}_{modelname}.pt"))
        net.load_state_dict(state_dict)
        net=net.module
        net.to(device)
        net.eval()
        print(f"Test {save_iter}")
        length = len(dataset)
        total = 0
        count = 0
        loss_fn = torch.nn.MSELoss()
        with torch.no_grad():
            with open(os.path.join(load_path, f"{save_name}/{save_iter}.log"), 'w') as outfile:
                outfile.write("subjcet,name,pre,label\n")
                for j, data in enumerate(dataset):
                    data["faceImg"] = data["face"].to(device)
                    labels = data["label"].to(device)
                    print(labels)
                    print(labels.shape)
                    output = net(data['faceImg'])
                    pred = nn.Softmax(dim=1)(output)
                    names = "name"
                    print(f'\r[Batch : {j}]', end='')
                    #print(f'gazes: {gazes.shape}')
                    for k, pre in enumerate(pred.argmax(1)):
                        # print(pre)
                        # print(labels[k])
                        # print("----------")
                        total += 1
                        if pre == labels[k]:
                            count += 1
                            # print("预测正确")
                        # else:
                            # print("预测错误")
                        
                        log = ["name" , str(pre) , str(labels[k])]
                        
                        outfile.write(",".join(log) + "\n")

                loger = f"[{save_iter}] Total Num: {count}, avg: {count/total} \n"
                outfile.write(loger)
                epoch_log.write(loger)
                print(loger)

