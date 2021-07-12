# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:23:48 2019

@author: richard
"""

def openpose_parameter():
    params = dict()
    params["model_folder"] = "/home/richard/Documents/openpose/models/"
    # ===== Uncommand or activate the parameter For Real-Time Prediction =====     
    params["render_pose"] = 0
    params["display"] = 0
    params["num_gpu"] = -1
    params["num_gpu_start"] = 0
    # ========================================================================
    return params


raw_directory = "0_raw_video_security_check"

posture_feature_directory = "1_posture_features_extraction"

model_directory = "2_Model"