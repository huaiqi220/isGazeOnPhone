import model2
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import yaml
import math
import time
import importlib
from tensorboardX import SummaryWriter

if __name__ == "__main__":
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    config = yaml.load(open("config.yaml"), Loader=yaml.FullLoader)
    readername = config["reader"]
    dataloader = importlib.import_module(readername)
    config = config["train"]
    imagepath = config["data"]["image"]
    labelpath = config["data"]["label"]
    modelname = config["save"]["model_name"]
    writer = SummaryWriter('runs/resnet18_rot_data2')
    step_number = 0

    trains = os.listdir(labelpath)
    trains.sort()
    print(f"Train Sets Num:{len(trains)}")

    trainlabelpath = [os.path.join(labelpath, j) for j in trains] 

    now_path = "bs_"  + str(config["params"]["batch_size"]) + "_fix_lr_" + str(str(config["params"]["lr"]))  + "_ep_" + str(str(config["params"]["epoch"])) +"_rot_data2"


    save_path = os.path.join(config["save"]["save_path"],"checkpoint", now_path)


    # save_path = os.path.join(config["save"]["save_path"], "checkpoint")
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # print("Read data")
    # dataset = reader.txtload(path, "train", config["params"]["batch_size"], shuffle=True,
    #                          num_workers=0)
    print("Read data")
    dataset = dataloader.txtload(trainlabelpath, imagepath, config["params"]["batch_size"], shuffle=True, num_workers=0, header=True)

    print("Model building")
    # net = model.ResNet34(2).to(device)
    net = model2.Resnet(model2.basic_block,[2,2,2,2],2)

    net.train()
    net = nn.DataParallel(net,device_ids=[4])
    # state_dict = torch.load(os.path.join("/home/work/didonglin/GazeTR/new-code/model-code/aff-net/AFF-Net-main/checkpoint", f"Iter_12_AFF-Net.pt"))
    # net.load_state_dict(state_dict)
    net.to(device)

    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    base_lr = config["params"]["lr"]
    cur_step = 0
    decay_steps = config["params"]["decay_step"]
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), base_lr,
                                weight_decay=0.0005)
    print("Traning")
    length = len(dataset)
    cur_decay_index = 0
    with open(os.path.join(save_path, "train_log"), 'w') as outfile:
        for epoch in range(1, config["params"]["epoch"] + 1):
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                base_lr = base_lr * config["params"]["decay"]
                cur_decay_index = cur_decay_index + 1
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr
            #if (epoch <= 10):
            #    continue
            time_begin = time.time()
            for i, (data) in enumerate(dataset):

                data["faceImg"] = data["face"].to(device)
                label = data["label"].to(device)
                # print(label)
                # print(label.shape)
                pre = net( data['faceImg'])
                loss = loss_fn(pre,label.type(torch.LongTensor).to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                time_remain = (length-i-1) * ((time.time()-time_begin)/(i+1)) /  3600   #time estimation for current epoch
                epoch_time = (length-1) * ((time.time()-time_begin)/(i+1)) / 3600       #time estimation for 1 epoch
                #person_time = epoch_time * (config["params"]["epoch"])                  #time estimation for 1 subject
                time_remain_total = time_remain + \
                                    epoch_time * (config["params"]["epoch"]-epoch)
                                    #person_time * (len(subjects) - subject_i - 1) 
                writer.add_scalar('loss',loss,global_step=step_number)
                step_number = step_number + 1
                log = f"[{epoch}/{config['params']['epoch']}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr} time:{time_remain:.2f}h total:{time_remain_total:.2f}h"
                outfile.write(log + "\n")
                # if i % 20 == 0:
                print(log)
                sys.stdout.flush()
                outfile.flush()

            if epoch % config["save"]["step"] == 0:
                torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{modelname}.pt"))

