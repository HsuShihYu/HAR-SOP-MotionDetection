# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 19:27:33 2019

@author: richard
"""

import sequence_global_variable as sgv
import os
import cv2
import numpy as np
import math
from openpose import pyopenpose as op
import csv

input_dir = sgv.raw_directory
output_dir = sgv.posture_feature_directory

opWrapper = op.WrapperPython()
opWrapper.configure(sgv.openpose_parameter())
opWrapper.start()

actions = [dI for dI in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir,dI))]

motion_class = 0

posture_features_all_video = []

for action in actions :
    
    path_dir = '{}/{}'.format(input_dir, action)
    data = os.listdir(path_dir)
    
    posture_feature = []
    
    for video_list in data :
        
        path_file = '{}/{}'.format(path_dir, video_list)    
        capture_read = cv2.VideoCapture(path_file)
        
        width = int(capture_read.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture_read.get(cv2.CAP_PROP_FPS)
        
        frames = []
        point_frames = []
        count_frame = 0
        
        while(capture_read.isOpened()) :
            
            res, frame = capture_read.read()
            
            if res == False :
                break
            else :
                datum = op.Datum()               
                datum.cvInputData = frame
                opWrapper.emplaceAndPop([datum])
                
                keypoints = datum.poseKeypoints
                output_image = datum.cvOutputData
                
                if (keypoints.shape != ()):
                    frames.append(output_image)
                    point_frames.append(keypoints)
                
                    count_frame = count_frame + 1
        
        frames = np.array(frames)
        
        feature_value = np.zeros((count_frame, 26))
               
        for i in range(count_frame):
                            
            if(point_frames[i][0][1][0] - point_frames[i][0][8][0] != 0 or point_frames[i][0][1][1] - point_frames[i][0][8][1] != 0) :
                feature_value[i][0] = ((point_frames[i][0][3][0] - point_frames[i][0][1][0]) + (point_frames[i][0][3][1] - point_frames[i][0][1][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)
                feature_value[i][1] = ((point_frames[i][0][4][0] - point_frames[i][0][1][0]) + (point_frames[i][0][4][1] - point_frames[i][0][1][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)
                feature_value[i][2] = ((point_frames[i][0][6][0] - point_frames[i][0][1][0]) + (point_frames[i][0][6][1] - point_frames[i][0][1][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)
                feature_value[i][3] = ((point_frames[i][0][7][0] - point_frames[i][0][1][0]) + (point_frames[i][0][7][1] - point_frames[i][0][1][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)
                feature_value[i][4] = ((point_frames[i][0][10][0] - point_frames[i][0][4][0]) + (point_frames[i][0][10][1] - point_frames[i][0][4][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)
                feature_value[i][5] = ((point_frames[i][0][13][0] - point_frames[i][0][7][0]) + (point_frames[i][0][13][1] - point_frames[i][0][7][1])) / math.sqrt((point_frames[i][0][1][0] - point_frames[i][0][8][0])**2 + (point_frames[i][0][1][1] - point_frames[i][0][8][1])**2)                                  
            
            vector_x_1 = point_frames[i][0][4][0] - point_frames[i][0][3][0]
            vector_x_2 = point_frames[i][0][2][0] - point_frames[i][0][3][0]
            vector_y_1 = point_frames[i][0][4][1] - point_frames[i][0][3][1]
            vector_y_2 = point_frames[i][0][2][1] - point_frames[i][0][3][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][6] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = point_frames[i][0][7][0] - point_frames[i][0][6][0]
            vector_x_2 = point_frames[i][0][5][0] - point_frames[i][0][6][0]
            vector_y_1 = point_frames[i][0][7][1] - point_frames[i][0][6][1]
            vector_y_2 = point_frames[i][0][5][1] - point_frames[i][0][6][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][7] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = point_frames[i][0][3][0] - point_frames[i][0][1][0]
            vector_x_2 = point_frames[i][0][8][0] - point_frames[i][0][1][0]
            vector_y_1 = point_frames[i][0][3][1] - point_frames[i][0][1][1]
            vector_y_2 = point_frames[i][0][8][1] - point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][8] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = point_frames[i][0][6][0] - point_frames[i][0][1][0]
            vector_x_2 = point_frames[i][0][8][0] - point_frames[i][0][1][0]
            vector_y_1 = point_frames[i][0][6][1] - point_frames[i][0][1][1]
            vector_y_2 = point_frames[i][0][8][1] - point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][9] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = point_frames[i][0][4][0] - point_frames[i][0][1][0]
            vector_x_2 = point_frames[i][0][8][0] - point_frames[i][0][1][0]
            vector_y_1 = point_frames[i][0][4][1] - point_frames[i][0][1][1]
            vector_y_2 = point_frames[i][0][8][1] - point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][10] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = point_frames[i][0][7][0] - point_frames[i][0][1][0]
            vector_x_2 = point_frames[i][0][8][0] - point_frames[i][0][1][0]
            vector_y_1 = point_frames[i][0][7][1] - point_frames[i][0][1][1]
            vector_y_2 = point_frames[i][0][8][1] - point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][11] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            for j in range(12, 18):
                for k in range(6, 12):
                    if (feature_value[i][k] > 330 and feature_value[i][k] <= 360) or (feature_value[i][k] >= 0 and feature_value[i][k] <= 30):
                        feature_value[i][j] = float(1)/float(6)
                    elif (feature_value[i][k] > 30 and feature_value[i][k] <= 90):
                        feature_value[i][j] = float(2)/float(6)
                    elif (feature_value[i][k] > 90 and feature_value[i][k] <= 150):
                        feature_value[i][j] = float(3)/float(6)
                    elif (feature_value[i][k] > 150 and feature_value[i][k] <= 210):
                        feature_value[i][j] = float(4)/float(6)
                    elif (feature_value[i][k] > 210 and feature_value[i][k] <= 270):
                        feature_value[i][j] = float(5)/float(6)
                    else:
                        feature_value[i][j] = 1
            
                        
            feature_value[i][6] = float(feature_value[i][6])/float(360)
            feature_value[i][7] = float(feature_value[i][7])/float(360)
            feature_value[i][8] = float(feature_value[i][8])/float(360)
            feature_value[i][9] = float(feature_value[i][9])/float(360)
            feature_value[i][10] = float(feature_value[i][10])/float(360)
            feature_value[i][11] = float(feature_value[i][11])/float(360)
            
            # Array (18, 19) for Human position(Front or Back); Array(20, 21, 22, 23, 24) for Hand position (Up, Right, Down, Left, middle) 
            if (action == "Cleaning_Head"):
                # position Human = > Front
                feature_value[i][18] = 1
                # Hand position => Up
                feature_value[i][20] = 1
                
                feature_value[i][25] = 0
            elif (action == "Cleaning_Back_Head"):
                # position Human = > Back
                feature_value[i][19] = 1
                # Hand position => Up
                feature_value[i][20] = 1
                
                feature_value[i][25] = 1
            elif (action == "Cleaning_Left_Chest_Hand"):
                # position Human = > Front
                feature_value[i][18] = 1
                # Hand position => left
                feature_value[i][23] = 1
                
                feature_value[i][25] = 2
            elif (action == "Cleaning_Right_Chest_Hand"):
                # position Human = > Front
                feature_value[i][18] = 1
                # Hand position => right
                feature_value[i][21] = 1
                
                feature_value[i][25] = 3
            elif (action == "Cleaning_Back"):
                # position Human = > Middle
                feature_value[i][19] = 1
                # Hand position => down
                feature_value[i][24] = 1
                
                feature_value[i][25] = 4
            elif (action == "Cleaning_Legs"):
                # position Human = > Front
                feature_value[i][18] = 1
                # Hand position => down
                feature_value[i][22] = 1
                
                feature_value[i][25] = 5
            elif (action == "Cleaning_Back_Legs"):
                # position Human = > Back
                feature_value[i][19] = 1
                # Hand position => down
                feature_value[i][22] = 1
                
                feature_value[i][25] = 6
                
            
        posture_feature.append(feature_value) 
           
        filename_output = video_list.split('.')[0]
#        posture_feature_directory = '{}/{}'.format(output_dir, action)
#        
#        if not os.path.exists(posture_feature_directory):
#            os.makedirs(posture_feature_directory)
#        
#        fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#        
#        capture_write = cv2.VideoWriter('{}/{}.avi'.format(posture_feature_directory, filename_output), fourcc, fps, (width, height))
#        
#        iter_frame = 0
#        while iter_frame < count_frame :
#            capture_write.write(frames[iter_frame])
#            iter_frame += 1
        
        print('{}/{}.avi'.format(action, filename_output))
        
        capture_read.release()
#        capture_write.release()
        
    posture_features_all_video.append(posture_feature)
    
                    