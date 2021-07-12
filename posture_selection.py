# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 14:01:58 2019

@author: richard
"""

import numpy as np
import posture_features_extraction as pfe
from sklearn.cluster import KMeans

posture_features_all_video = pfe.posture_features_all_video

x = len(posture_features_all_video)
centroid_features = []
class_features = []

for i in range(x):
    
    y = len(posture_features_all_video[i])
    class_videos = np.zeros(7)
    
    
    for j in range(y):
        
        kmeans = KMeans(n_clusters=30, random_state=0).fit(posture_features_all_video[i][j])
        class_videos[int(kmeans.cluster_centers_[0, 25])] = 1        
        
        centroid_features.append(kmeans.cluster_centers_[:, 0:25])
        class_features.append(class_videos)
        
        