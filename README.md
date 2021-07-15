# HAR-SOP-MotionDetection
 Used Deep Learning to detect self-cleaning activities of operators in real-time before they enter the Dust-free Room
 View the [Project Report](https://drive.google.com/uc?export=download&id=1XpMkNKQq06au4DEoWKEoOaOlKq_2mWbn) of MOST College Student Research

## Dataset

[0_raw_video_check_security_check.zip](https://drive.google.com/uc?export=download&id=1rMqTzyLOiVlXNxiOHZXGDOJk1GkkI_Am)  
  
There are a total of 1015 videos (MPEG4), with each category having 145 videos.  
All the videos were captured at 515 frames. Each video has a spatial resolution of 1920x1080 pixels.  

[1_posture_feature_extraction.zip](https://drive.google.com/uc?export=download&id=1vHI8d2Hln6iX1rl-JXq3QDcAcilt1fvX)  
The output of extracting posture feature from above videos.   

## Requirement
1. Language = `Python3.6x`, OS = `Ubuntu 16.04`
3. Install `Darknet` from [Darknet Installation](https://pjreddie.com/darknet/install/)
4. Install `CUDA 9.2` to use GPU trainning from [CUDA Installation](https://developer.nvidia.com/cuda-downloads)
5. Install `TensorFlow 1.8.0 GPU version` from [TensorFlow Installation](https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-9-2-for-python-on-ubuntu/?fbclid=IwAR2juxu_RnKGk5ZzDAuMc2RvgFAFs7uL8ga0meqXnlO2DmoDbaB31grm77I)
6. Install `OpenPose` from [OpenPose Installation](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation)
7. Install `OpenCV` from [OpenCV Installation](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html)





