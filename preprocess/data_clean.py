"""
?
"""
import os


file_path = "/disk1/repository/isGazeOnPhone/"
on_video = os.path.join(file_path, "OnScreen")
not_video = os.path.join(file_path, "NotOnScreen")

count_video = 0
on_videos = os.listdir(on_video)
for ovideo in on_videos:
    ovp = os.path.join(on_video, ovideo)
    count_video = count_video + 1
    print('正在处理' + ovp)
    print(count_video)
    print("---------------------")

n_videos = os.listdir(not_video)
for nvideo in n_videos:
    nvp = os.path.join(not_video, nvideo)
    count_video = count_video + 1
    print('正在处理' + nvp)
    print(count_video)
    print("---------------------")
