# HAR-SOP-MotionDetection
 Used Deep Learning to detect self-cleaning activities of operators in real-time before they enter the Dust-free Room   
 View the [Project Report](https://drive.google.com/uc?export=download&id=1mo618A_-mqqq6D8m5c5nl8jJPJqDvZZ_) of MOST College Student Research  
 
 

## Dataset

[0_raw_video_check_security_check.zip](https://drive.google.com/uc?export=download&id=1rMqTzyLOiVlXNxiOHZXGDOJk1GkkI_Am)  
  
There are a total of 1015 videos (MPEG4), with each category having 145 videos.  
All the videos were captured at 515 frames. Each video has a spatial resolution of 1920x1080 pixels.  

[1_posture_feature_extraction.zip](https://drive.google.com/uc?export=download&id=1vHI8d2Hln6iX1rl-JXq3QDcAcilt1fvX)  
The output of extracting posture feature from above videos.   

## Requirement
`Python3.6x`(preferably from the [Anaconda Distribution](https://www.anaconda.com/download/))
Linux version preferably using `Ubuntu 16.04`  

Tools/Libraries we need to install on the machine:
1. [Darknet](https://pjreddie.com/darknet/install/)
2. [CUDA 9.2](https://developer.nvidia.com/cuda-downloads) for GPU training
3. [TensorFlow 1.8.0 GPU version](https://www.pytorials.com/how-to-install-tensorflow-gpu-with-cuda-9-2-for-python-on-ubuntu/?fbclid=IwAR2juxu_RnKGk5ZzDAuMc2RvgFAFs7uL8ga0meqXnlO2DmoDbaB31grm77I)
4. [OpenCV](https://docs.opencv.org/4.5.2/d7/d9f/tutorial_linux_install.html)
5. [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose#installation)

