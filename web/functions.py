import pandas as pd
import numpy as np

import json
import os
import random
from collections import Counter
from itertools import islice
import operator

from keras.preprocessing.image import ImageDataGenerator, load_img, image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import applications
from tensorflow.keras.models import load_model

from skimage.color import rgb2lab, deltaE_cie76
from sklearn.cluster import KMeans
import cv2

model = load_model('flaskstylematch/static/models/model.h5')

def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))

def get_colors(image, number_of_colors, show_chart):
    # Read in image and convert to RGB
    image = cv2.imread(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    # Cluster the pixels by color using kmeans
    clusters = KMeans(n_clusters = number_of_colors, random_state = 3)
    labels = clusters.fit_predict(modified_image)
        
    # Count the number of pixels labeled by each cluster
    counts = Counter(labels)
    counts = dict(sorted(counts.items()))
    
    # Get the center of each cluster in RGB space
    center_colors = clusters.cluster_centers_
    
    # Put center colors in same order as counts
    # Convert center colors to Hex and save in hex and rgb lists
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    lab_colors = [rgb2lab(np.uint8(np.asarray([[rgb_colors[i]]]))) for i in range(number_of_colors)]

    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
        
    # Store values in dataframe to return 
    image_colors_dict = {'percent': [], 'hex': [], 'R': [], 'G': [], 'B': [], 'Lab': []}
    
    total_pixels = sum(counts.values())
    for index, value in enumerate(counts.values()):
        image_colors_dict['percent'].append((value/total_pixels)*100)
        image_colors_dict['hex'].append(hex_colors[index])
        image_colors_dict['R'].append(rgb_colors[index][0])
        image_colors_dict['G'].append(rgb_colors[index][1])
        image_colors_dict['B'].append(rgb_colors[index][2])
        image_colors_dict['Lab'].append(lab_colors[index])
        
            
    image_colors_df = pd.DataFrame(data = image_colors_dict)
    
    return image_colors_df

def match_colors(chair_df, rug_color_dict):   
    # Get Lab for dominant color in chair that is not close to the background
    
    # Get lab for background
    background_lab = rgb2lab(np.uint8(np.asarray([[[255, 255, 255]]])))
    
    # Convert chair dataframe to dictionary
    chair_color_dict = chair_df.to_dict()
    
    # Compare background to colors in chair and find closest match
    distance_from_background = [deltaE_cie76(lab, background_lab)[0][0] for lab in chair_color_dict['Lab'].values()]    
    background_index = np.argmin(distance_from_background)
    
    # Get index of dominant color in chair other than the background
    color_percentages = chair_color_dict['percent']
    color_percentages.pop(background_index)
    dominant_color_index = max(color_percentages.items(), key=operator.itemgetter(1))[0]
    dominant_color_lab = chair_color_dict['Lab'][dominant_color_index]
        
    # For each rug find the closest color to the dominant color in the accent chair
    # Setup color distance dictionary
    color_distance_dict = {}

    # For each rug loop through all the colors in that rug, store the closest color in the color_distance_dict
    for index, rug in enumerate(rug_color_dict.keys()):
        temp_distance_dict = {}
        temp_distance_to_background = {}
        for color in rug_color_dict[rug].keys():
            rug_lab = [[[rug_color_dict[rug][color]['lab_1'], rug_color_dict[rug][color]['lab_2'],
                        rug_color_dict[rug][color]['lab_3']]]]
            distance = deltaE_cie76(dominant_color_lab, rug_lab)[0][0]
            temp_distance_dict[color] = distance
        closest_color = min(temp_distance_dict.items(), key=operator.itemgetter(1))[0]
        color_distance_dict[rug] = []
        color_distance_dict[rug].append(closest_color)
        color_distance_dict[rug].append(rug_color_dict[rug][closest_color]['percent'])
        color_distance_dict[rug].append(temp_distance_dict[closest_color])

    # Order rugs by closeness to dominant chair color of closest color in rug
    color_distance_dict_sorted = {k: v for k, v in sorted(color_distance_dict.items(), key=lambda item: item[1][2])}

    top_rugs_dict = {}
    for rug in color_distance_dict_sorted:
        if color_distance_dict_sorted[rug][1] > 10:
            if len(top_rugs_dict.keys()) < 4:
                top_rugs_dict[rug] = 'match'
            else:
                break

    return top_rugs_dict

def filter_style(rug_df, style):

    # Convert rug dataframe to dictionary with rug filename - hex - percent, lab
    rug_color_dict = {}
    for rug, colors in rug_df.groupby('rug_filename'):
        if style in rug:
            rug_color_dict[rug] = {}
            for index, row in colors.iterrows():
                rug_color_dict[rug][row['hex']] = {}
                rug_color_dict[rug][row['hex']]['percent'] = row['percent']
                rug_color_dict[rug][row['hex']]['lab_1'] = row['lab_1']
                rug_color_dict[rug][row['hex']]['lab_2'] = row['lab_2']
                rug_color_dict[rug][row['hex']]['lab_3'] = row['lab_3']

    return rug_color_dict

def get_rugs(style, filename):

    # Get dominant color of chair
    chair_colors_df = get_colors('flaskstylematch/static/images/' + filename, 7, False)

    # Load rug colors
    rug_colors_df = pd.read_csv('flaskstylematch/static/rug_data/rug_colors.csv')

    # Reduce rug_colors_df to only the matching style
    filtered_rug_color_dict = filter_style(rug_colors_df, style)

    # Find rugs that match the dominant color of the chair
    top_rugs_dict = match_colors(chair_colors_df, filtered_rug_color_dict)

    print(top_rugs_dict)

    first = list(top_rugs_dict.keys())[0]
    second = list(top_rugs_dict.keys())[1]
    third = list(top_rugs_dict.keys())[2]

    houzz_site = 'https://www.houzz.com/products/'

    first_link = houzz_site + first.split('_')[0]
    second_link = houzz_site + second.split('_')[0]
    third_link = houzz_site + third.split('_')[0]


    return first, second, third, first_link, second_link, third_link


def predict(image_upload):

    # img_width, img_height = 150, 150
    img_width, img_height = 224, 224

    #load image you want to make prediction for
    img = image.load_img(image_upload, target_size = (img_width, img_height))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255

    pred = model.predict(img_tensor)
    index_predict = np.argmax(pred[0])

    dict_labels = {0: 'Contemporary', 1: 'Midcentury', 2: 'Traditional', 3: 'Transitional'}
    
    return dict_labels[index_predict]




