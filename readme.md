# is Gaze On Phone?

Judge whether the user is looking at the smartphone screen through head posture recognition.

PaddlePaddle and Paddle detection kit need to be installed


## runs
python code/interface_paddle_detection.py 
--snapshot "code/hopenet_robust_alpha1.pkl" 
--face_model "mmod_human_face_detector.dat" 
--video "2.mp4" 
--output_string "test1" 
--n_frames 1890 
--fps 30
