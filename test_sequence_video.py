# -*- coding: utf-8 -*-
"""
Created on Fri May  3 21:49:12 2019

@author: richard
"""

import sequence_global_variable as sgv
import cv2
import numpy as np
import math
import copy
from openpose import pyopenpose as op
from sklearn.cluster import KMeans
from keras.models import model_from_json
from keras import backend as K
import darknet
import time

opWrapper = op.WrapperPython()
opWrapper.configure(sgv.openpose_parameter())
opWrapper.start()
model_path = sgv.model_directory

net = darknet.load_net("/home/richard/Documents/darknet/SOP_cfg_Updated/yolov3-tiny.cfg", "/home/richard/Documents/darknet/SOP_result_Updated/yolov3-tiny_best.weights", 0)
meta = darknet.load_meta("/home/richard/Documents/darknet/SOP_cfg_Updated/obj.data")

configPath = "/home/richard/Documents/darknet/SOP_cfg_Updated/yolov3-tiny.cfg"
weightPath = "/home/richard/Documents/darknet/SOP_result_Updated/yolov3-tiny_best.weights"
netMain = darknet.load_net_custom(configPath.encode("ascii"), weightPath.encode("ascii"), 0, 1)
darknet_image = darknet.make_image(darknet.network_width(netMain), darknet.network_height(netMain),3)

video_filepath = "/home/richard/Documents/TryDeepLearning/SequenceAndMultiObject_Dataset_SOP/Sequence_8.avi"

capture_read = cv2.VideoCapture(video_filepath)

width = int(capture_read.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(capture_read.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = capture_read.get(cv2.CAP_PROP_FPS)

class_index = ["Cleaning_Head", "Cleaning_Back_Head", "Cleaning_Left_Chest_Hand", "Cleaning_Right_Chest_Hand", "Cleaning_Back", "Cleaning_Legs", "Cleaning_Back_Legs"]
count_activity_frame = {}

# Initalization count_activity_frame
for i in range(len(class_index)):
    count_activity_frame[class_index[i]] = 0

flag_class = ""
flag_class_before = ""
flag_video = False

frames = []
point_frames = []
prediction_frame = []
count_frame = 0

complete_seq_font = cv2.FONT_HERSHEY_SIMPLEX
complete_seq_location = (85, 160)
complete_seq_scale = 1
complete_seq_color = (202, 52, 51)
complete_seq_line = 3

motion_class= "-"

index_asd = 0

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
        
        frames.append(frame)
        point_frames.append(keypoints)
        
        count_frame += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (darknet.network_width(netMain), darknet.network_height(netMain)), interpolation=cv2.INTER_LINEAR)
        darknet.copy_image_from_bytes(darknet_image, frame_resized.tobytes())

        r = darknet.detect(net, meta, darknet_image, 0.87)

        if (r):
            if(flag_class == ""):
                count_activity_frame[r[0][0]] += 1
                flag_class = r[0][0]
                flag_class_before = r[0][0]
            else:
                if (flag_class != r[0][0]):
                    if (count_activity_frame[flag_class] > 60):
                        flag_video = True
                        flag_class_before = flag_class
                        print("%s : %d, Flag video:%d\n" %(flag_class, count_activity_frame[flag_class], flag_video))
                    
                    flag_class = r[0][0]

                count_activity_frame[r[0][0]] += 1

    cv2.putText(frame, 'Motion Class : {}'.format(motion_class), complete_seq_location, complete_seq_font, complete_seq_scale, complete_seq_color, complete_seq_line)
    prediction_frame.append(frame)    
    cv2.imshow('frame', cv2.resize(frame, (1920/2, 1080/2)))
        
    if ( cv2.waitKey(10) & 0xFF == 27):
        break   
    
    if (flag_video == True):
        start_time = time.time()
        print("%s : %d, Current Frame: %d\n" %(flag_class_before, count_activity_frame[flag_class_before], count_frame))      
        
        copy_point_frames = []
        copy_point_frames = copy.copy(point_frames[count_frame - count_activity_frame[flag_class_before]: count_frame-1])
        
        feature_value = np.zeros((count_activity_frame[flag_class_before], 25))
        
        for i in range(count_activity_frame[flag_class_before]-1):
            
            if(copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0] != 0 or copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1] != 0) :
                feature_value[i][0] = ((copy_point_frames[i][0][3][0] - copy_point_frames[i][0][1][0]) + (copy_point_frames[i][0][3][1] - copy_point_frames[i][0][1][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)
                feature_value[i][1] = ((copy_point_frames[i][0][4][0] - copy_point_frames[i][0][1][0]) + (copy_point_frames[i][0][4][1] - copy_point_frames[i][0][1][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)
                feature_value[i][2] = ((copy_point_frames[i][0][6][0] - copy_point_frames[i][0][1][0]) + (copy_point_frames[i][0][6][1] - copy_point_frames[i][0][1][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)
                feature_value[i][3] = ((copy_point_frames[i][0][7][0] - copy_point_frames[i][0][1][0]) + (copy_point_frames[i][0][7][1] - copy_point_frames[i][0][1][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)
                feature_value[i][4] = ((copy_point_frames[i][0][10][0] - copy_point_frames[i][0][4][0]) + (copy_point_frames[i][0][10][1] - copy_point_frames[i][0][4][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)
                feature_value[i][5] = ((copy_point_frames[i][0][13][0] - copy_point_frames[i][0][7][0]) + (copy_point_frames[i][0][13][1] - copy_point_frames[i][0][7][1])) / math.sqrt((copy_point_frames[i][0][1][0] - copy_point_frames[i][0][8][0])**2 + (copy_point_frames[i][0][1][1] - copy_point_frames[i][0][8][1])**2)                                  
            
            vector_x_1 = copy_point_frames[i][0][4][0] - copy_point_frames[i][0][3][0]
            vector_x_2 = copy_point_frames[i][0][2][0] - copy_point_frames[i][0][3][0]
            vector_y_1 = copy_point_frames[i][0][4][1] - copy_point_frames[i][0][3][1]
            vector_y_2 = copy_point_frames[i][0][2][1] - copy_point_frames[i][0][3][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][6] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = copy_point_frames[i][0][7][0] - copy_point_frames[i][0][6][0]
            vector_x_2 = copy_point_frames[i][0][5][0] - copy_point_frames[i][0][6][0]
            vector_y_1 = copy_point_frames[i][0][7][1] - copy_point_frames[i][0][6][1]
            vector_y_2 = copy_point_frames[i][0][5][1] - copy_point_frames[i][0][6][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][7] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = copy_point_frames[i][0][3][0] - copy_point_frames[i][0][1][0]
            vector_x_2 = copy_point_frames[i][0][8][0] - copy_point_frames[i][0][1][0]
            vector_y_1 = copy_point_frames[i][0][3][1] - copy_point_frames[i][0][1][1]
            vector_y_2 = copy_point_frames[i][0][8][1] - copy_point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][8] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = copy_point_frames[i][0][6][0] - copy_point_frames[i][0][1][0]
            vector_x_2 = copy_point_frames[i][0][8][0] - copy_point_frames[i][0][1][0]
            vector_y_1 = copy_point_frames[i][0][6][1] - copy_point_frames[i][0][1][1]
            vector_y_2 = copy_point_frames[i][0][8][1] - copy_point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][9] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = copy_point_frames[i][0][4][0] - copy_point_frames[i][0][1][0]
            vector_x_2 = copy_point_frames[i][0][8][0] - copy_point_frames[i][0][1][0]
            vector_y_1 = copy_point_frames[i][0][4][1] - copy_point_frames[i][0][1][1]
            vector_y_2 = copy_point_frames[i][0][8][1] - copy_point_frames[i][0][1][1]
            
            dot = vector_x_1*vector_x_2 + vector_y_1*vector_y_2
            det = vector_x_1*vector_y_2 - vector_y_1*vector_x_2
            
            feature_value[i][10] = (math.degrees(math.atan2(det, dot)) + 360)%360
            
            vector_x_1 = copy_point_frames[i][0][7][0] - copy_point_frames[i][0][1][0]
            vector_x_2 = copy_point_frames[i][0][8][0] - copy_point_frames[i][0][1][0]
            vector_y_1 = copy_point_frames[i][0][7][1] - copy_point_frames[i][0][1][1]
            vector_y_2 = copy_point_frames[i][0][8][1] - copy_point_frames[i][0][1][1]
            
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
            
            if (flag_class_before == "Cleaning_Head"):
                feature_value[i][18] = 1
                feature_value[i][20] = 1
            elif (flag_class_before == "Cleaning_Back_Head"):
                feature_value[i][19] = 1
                feature_value[i][20] = 1
            elif (flag_class_before == "Cleaning_Left_Chest_Hand"):
                feature_value[i][18] = 1
                feature_value[i][23] = 1
            elif (flag_class_before == "Cleaning_Right_Chest_Hand"):
                feature_value[i][18] = 1
                feature_value[i][21] = 1
            elif (flag_class_before == "Cleaning_Back"):
                feature_value[i][19] = 1
                feature_value[i][24] = 1
            elif (flag_class_before == "Cleaning_Legs"):
                feature_value[i][18] = 1
                feature_value[i][22] = 1
            elif (flag_class_before == "Cleaning_Back_Legs"):
                feature_value[i][19] = 1
                feature_value[i][22] = 1

        #Group The Frame
        kmeans = KMeans(n_clusters=30, random_state=0).fit(feature_value)
        testing_data = kmeans.cluster_centers_[:, :]
        testing_data = np.reshape(testing_data, (1, 30, 25))

        json_file = open("{}/model_300_GRU.json".format(model_path), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights("{}/model_weight_300_GRU.h5".format(model_path))

        predict = loaded_model.predict(testing_data)
        
        data_class_index = np.argmax(predict[0])
        if (predict[0, data_class_index] > 0.5):
            data_class_index += 1
        else:
            data_class_index = -1
            
        
        print("\n========================================================================")
        print(predict)
        if (data_class_index == 1):
            motion_class = "Cleaning Head"
            print("Prediction class is Cleaning Head")
        elif(data_class_index == 2):
            motion_class = "Cleaning Back Head"
            print("Prediction class is Cleaning Back Head")
        elif(data_class_index == 3):
            motion_class = "Cleaning Left Chest Hand"
            print("Prediction class is Cleaning Left Chest Hand")
        elif(data_class_index == 4):
            motion_class = "Cleaning Right Chest Hand"
            print("Prediction class is Cleaning Right Chest Hand")
        elif(data_class_index == 5):
            motion_class = "Cleaning Back"
            print("Prediction class is Cleaning Back")
        elif(data_class_index == 6):
            motion_class = "Cleaning Legs"
            print("Prediction class is Cleaning Legs")
        elif(data_class_index == 7):
            motion_class = "Cleaning Back Legs"
            print("Prediction class is Cleaning Back Legs")
        else:
            motion_class = "Please Repeat the Action"
            print("The motion class is not cleared")
        
        print("Execution Time : %s seconds" % (time.time() - start_time))
        print("========================================================================")

        count_activity_frame[flag_class_before] = 0        
        flag_video = False
        

#fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
#capture_write = cv2.VideoWriter('{}/Sequence_Video_Result.avi'.format(model_path), fourcc, fps, (width, height))
#iter_frame = 0
#while iter_frame < count_frame :
#    capture_write.write(prediction_frame[iter_frame])
#    iter_frame += 1
#    
#
#capture_write.release()
capture_read.release()
cv2.destroyAllWindows()

K.clear_session()    

