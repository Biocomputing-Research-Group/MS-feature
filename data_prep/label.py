import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import os

# this file generates the list of coordinates of bounding boxes in misc folder
# also generates an annotated image of in whole_img_labeled

# the top number of features to look for in the output by maxquant (we don't want to annotate the features that are not
# visible in the image)
# tune this constant so that there aren't visible features omitted or invisible features annotated
NUM_FEATURES = 900

files = os.listdir('whole_img')
print(files)
s = input("enter 'a' to process all, or the beginning index of the files to process: ")
if s != 'a':
    e = input('enter end index: (exclusive, if you want to process file at index 1 then enter 2) ')
    files = files[int(s):int(e)+1]

for file_name in files:
    tmp = cv2.imread('whole_img/'+file_name)
    file_name = file_name.split('.')[0]
    temp = np.load('misc/' + file_name + '_misc.npy')
    max_rt, min_rt, max_mz, min_mz, HEIGHT, WIDTH = temp[0], temp[1], temp[2], temp[3], tmp.shape[0], tmp.shape[1]

    print(f'visualizing...{file_name}')
    # each pixel in the 'new' 2d array has only one value specifying how white the pixel is, so 'new' is a black-and-white image
    # each pixel in the 'whole_img' array has three values (BGR), each representing how blue, green, and red the pixel is, so 'whole_img' is a colored image
    img = tmp

    def show_from_mq(mn, mx, high, low, WIDTH, HEIGHT, img):
        # read the output file in as dataframe
        dataframe = pd.read_csv('mq/' + file_name + '-features.txt', sep='\t')
        # leave only MS2 features
        # dataframe = dataframe[dataframe['MS/MS Count'] < 3]
        print(f"max rt={max([float(e['Retention time']) - float(e['Retention length'])/2 for _, e in dataframe.iterrows()])}")
        # print the minimum and maximum of mz values on the raw data
        print(mn, mx)
        # sort the rows of dataframe by intensity of the feature in descending order
        dataframe = dataframe.sort_values(by=['Intensity'], ascending=False)
        # find the minimum and maximum intensities
        max_inten, min_inten = dataframe.iloc[0]['Intensity'], dataframe.iloc[dataframe.shape[0] - 1]['Intensity']

        isotopic_distance_for_charge = [None, 1.0, 0.5, 0.33, 0.25, 0.17, 0.14, 0.13, 0.19]

        feature_bbox = np.zeros((NUM_FEATURES, 5))

        # draw a box on the image for the first NUM_FEATHERS features
        i = 0
        max_isotope = 0
        for _, row in tqdm(dataframe[:NUM_FEATURES].iterrows()):
            charge = int(row['Charge'])
            n_isotope = int(row['Number of isotopic peaks'])
            w = isotopic_distance_for_charge[charge] * (n_isotope-1)
            mz = float(row['m/z'])
            rt_mid, rt_len = float(row['Retention time']), float(row['Retention length'])
            # rt is a tuple of two values: the start and end retention times of the feature
            scale = 100
            rt = rt_mid + rt_len/(2*scale), rt_mid - rt_len/(2*scale)
            top_index = int(((rt[0] - low) / (high - low)) * (HEIGHT - 1))
            bot_index = int(((rt[1] - low) / (high - low)) * (HEIGHT - 1))
            # find where on the x axis this line should go
            left_index = int(((mz - 0.4 - mn) / (mx - mn)) * (WIDTH - 1))
            right_index = int(((mz + w - 0.4 - mn) / (mx - mn)) * (WIDTH - 1))
            # determine how bright this line should be
            relative_intensity = (float(row['Intensity']) - min_inten) / (max_inten - min_inten)
            # draw the line
            img = cv2.rectangle(img, (left_index, top_index), (right_index, bot_index), (150 + relative_intensity * (255 - 150), 90, 90), 1)
            feature_bbox[i] = [left_index, top_index, right_index, bot_index, row['Number of isotopic peaks']]
            max_isotope = max(max_isotope, row['Number of isotopic peaks'])
            i += 1
        print(max_isotope, file_name)
        return img, feature_bbox

    def show_from_dino(mn, mx, high, low, WIDTH, HEIGHT, img):
        file = pd.read_csv(file_name.split('.')[0] + ".features.tsv", sep="\t")
        print(mn, mx)
        file = file.sort_values(by=['intensityApex'], ascending=False)
        max_inten, min_inten = file.iloc[0]['intensityApex'], \
                               file.iloc[file.shape[0] - 1]['intensityApex']
        for i, row in tqdm(file.head(400).iterrows()):
            mz = float(row['mostAbundantMz'])
            rt = float(row['rtStart']), float(row['rtEnd'])
            top, bot = int(((rt[0] - low) / (high - low)) * (HEIGHT - 1)), int(
                ((rt[1] - low) / (high - low)) * (HEIGHT - 1))
            x = int(((mz - mn) / (mx - mn)) * (WIDTH - 1))
            relative_intensity = (row['intensityApex'] - min_inten) / (max_inten - min_inten)
            img = cv2.line(img, (x, top), (x, bot), (0, 150 + relative_intensity * (255 - 150), 0), 1)
        return img

    # read from the output by MaxQuant
    img = show_from_mq(min_mz, max_mz, max_rt, min_rt, WIDTH, HEIGHT, img)

    # LOOK HERE list of coordinates of bounding boxes are saved in misc
    np.save(f'misc/{file_name}_feature_bbox', img[1])
    img = img[0]
    img = read_from_dino(min_mz, max_mz, max_rt, min_rt, WIDTH, HEIGHT, img)

    # image status is true if 'whole_img' is successfully written onto the local folder, the 'cv2.imwrite' function writes the image
    print("image status:" + str(cv2.imwrite('whole_img_labeled/'+file_name + ".png", img)))
