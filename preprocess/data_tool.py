import cv2
import os
from cv2 import VideoCapture
import skvideo.io
import numpy as np

def saveAsPath(vide_path, image_path, kind, count_video, result_list):
    print('正在处理' + vide_path)
    print(count_video)
    situ = -1
    '''
    0注视1不注视
    '''
    if kind == "on":
        situ = 0

    if kind == "noton":
        situ = 1

    # label_outpath = os.path.join(os.path.join(label_path,mark),  str(count_video)  + ".label")
    # outfile = open(label_outpath, 'w')
    # outfile.write("Face_path Kind\n")

    capture = cv2.VideoCapture(vide_path)
    metadata = skvideo.io.ffprobe(vide_path)
    max = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    index = 0
    for i in range(max):
        '''
        index 用于划分测试集训练集
        条件判断用于掐头去尾以及裁剪
        '''
        if i > 180 and i < max - 180 and i % 10 == 0:
            index = index + 1
            capture.set(cv2.CAP_PROP_POS_FRAMES, flo1at(i))
            flag, frame = capture.read()
            if frame.shape[0] < frame.shape[1]:  # 获取视频自选择角度
                frame = np.rot90(frame)
                print("本视频经过旋转")
            # 读取每一帧，flag表示是否读取成功，frame为图片内容。
            fileName = kind + "_" + str(count_video) + "_" + str(i) + ".jpg"
            # print("正在保存：  " + fileName)
            if flag == True:
                cv2.imwrite(os.path.join(image_path, fileName), frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 100])
                label = " ".join([fileName, str(situ)])
                # outfile.write(label + "\n")
                if index % 5 == 0:
                    result_list[1].append(label)
                else:
                    result_list[0].append(label)

    capture.release()


# def getEveryImage(dir_path,output_path,kind):
#     '''
#     遍历文件夹，处理所有视频，抽成图像

#     '''
#     videos = os.listdir(dir_path)
#     count = 0
#     for video in videos:
#         saveAsPath(count,kind,os.path.join(dir_path,video),output_path)
#         count = count + 1


def getEveryImage(video_path, output_path):
    '''
    遍历文件夹，处理所有视频，抽成图像
    
    '''
    on_video = os.path.join(video_path, "on")
    not_video = os.path.join(video_path, "noton")

    image_path = os.path.join(output_path, "image")
    label_path = os.path.join(output_path, "label")

    test_label_path = os.path.join(label_path, "test")
    train_label_path = os.path.join(label_path, "train")

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if not os.path.exists(label_path):
        os.makedirs(label_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    count_video = 0
    '''
    count_video % 5 == 0归为测试集
    '''
    train_list = []
    test_list = []
    result_list = [train_list, test_list]

    on_videos = os.listdir(on_video)
    # print(len(on_videos))
    for ovideo in on_videos:
        ovp = os.path.join(on_video, ovideo)
        count_video = count_video + 1
        '''
        kind = "on" / "not on "
        '''
        saveAsPath(ovp, image_path, "on", count_video, result_list)

    noton_videos = os.listdir(not_video)
    for nvideo in noton_videos:
        nvp = os.path.join(not_video, nvideo)
        count_video = count_video + 1
        '''
        kind = "on" / "not on "
        '''
        saveAsPath(nvp, image_path, "noton", count_video, result_list)

    print("开始标签生成")
    train_label = os.path.join(train_label_path, "train" + ".label")
    outfile = open(train_label, 'w')
    outfile.write("Face_path Kind\n")
    for item in result_list[0]:
        outfile.write(item + "\n")

    outfile.close()

    test_label = os.path.join(test_label_path, "test" + ".label")
    outfile = open(test_label, 'w')
    outfile.write("Face_path Kind\n")
    for item in result_list[1]:
        outfile.write(item + "\n")

    outfile.close()

    print("所有视频处理完成")


if __name__ == "__main__":

    output_path = "/disk1/repository/isGazeOnPhone/new_data/output"
    video_path = "/disk1/repository/isGazeOnPhone/new_data"
    '''
    output_path:
        image:
        label:
            test:
            train:
    '''

    getEveryImage(video_path, output_path)

    # getEveryImage(dir_path,image_path,label_path,kind)
