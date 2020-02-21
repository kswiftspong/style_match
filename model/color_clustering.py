from sklearn.cluster import KMeans
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os
import pandas as pd
import operator
from itertools import islice
import json

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

def store_rug_colors(rug_dir):
    
    def listdir_nohidden(path):
        for f in os.listdir(path):
            if not f.startswith('.'):
                yield f
    rug_filenames = listdir_nohidden(rug_dir)
    
    # Setup dataframe
    rug_colors_df = pd.DataFrame(columns = ['rug_filename', 'percent', 'hex', 'R', 'G', 'B', 'Lab'])
    
    # Get main colors for each rug
    for rug in rug_filenames:
        temp_rug_colrs_df = get_colors(rug_dir + '/' + rug, 7, False)
        temp_rug_colrs_df['rug_filename'] = rug
        rug_colors_df = pd.concat([rug_colors_df, temp_rug_colrs_df], ignore_index=True, sort=False)
        
    return rug_colors_df

def save_rug_colors(rug_dir):
    rug_colors_df = store_rug_colors(rug_dir)

    rug_colors_df['lab_1'] = rug_colors_df.apply (lambda row: row['Lab'][0][0][0], axis=1)
    rug_colors_df['lab_2'] = rug_colors_df.apply (lambda row: row['Lab'][0][0][1], axis=1)
    rug_colors_df['lab_3'] = rug_colors_df.apply (lambda row: row['Lab'][0][0][2], axis=1)

    rug_colors_df.to_csv(r'../web/flaskexample/static/rug_data/rug_colors.csv')

if __name__ == "__main__":
	rug_dir = '../web/flaskexample/static/rugs/'
	save_rug_colors(rug_dir)


