"""
这个脚本从DGaze数据集均匀采样，补充数据到屏幕注视数据集中
同时生成标签
DGaze-5000: /disk2/repository/DGaze-5000/data_final/Image/
isGazeOnPhone: /disk2/repository/isGazeOnPhoneFull

"""

import os 
import shutil
import time


basePath = "/disk2/repository/DGaze-5000/data_final/Image/"
newPath = "/disk2/repository/isGazeOnPhoneFull"

label_path = os.path.join(newPath, "label", "train", "dgaze.label")
image_path = os.path.join(newPath, "image")
outfile = open(label_path, 'w')
outfile.write("Face_path Kind\n")

dirs_list = os.listdir(basePath)

for dir in dirs_list:
    current_path = os.path.join(basePath, dir, "full")
    image_list = os.listdir(current_path)
    image_list.sort()
    index = 5
    file_name = "photo_" + str(index) + ".jpg"
    image_copy_list = []
    while os.path.exists(os.path.join(current_path, file_name)) and index < 200: 
        image_copy_list.append(file_name)
        index = index + 5
        file_name = "photo_" + str(index) + ".jpg"
    
    """
    相片抽样完成，复制并生成标签于指定目录
    """
    # print(image_copy_list)
    # time.sleep(1000)
    for image in image_copy_list:
        target_name = str(dir) + "_" + image
        shutil.copyfile(os.path.join(current_path, image), os.path.join(image_path, target_name))
        # time.sleep(1000)
        outfile.write(target_name + " 0" + "\n")

outfile.close()




        


