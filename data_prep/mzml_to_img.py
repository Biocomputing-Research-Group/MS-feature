import sys
from decoder import *
import numpy as np
import time
from tqdm import tqdm
import cv2
# import matplotlib.pyplot as plt
import os
import math

files = os.listdir('src')
print(files)
s = input("enter 'a' to process all, or index of file to process particular one: ")
if s != 'a':
    e = input('enter end index: ')
    files = files[int(s):int(e)+1]

for file_name in files:
    file_path = 'src/'+file_name
    t = time.time()
    print(f'starting to process file {file_path}')

    # decode the xml file and store the information in 'result', a list of 'spectrum' objects
    listMz, listIn, listRt = decode(file_path)

    # setting width and height, this can be any number we set

    print('initializing min and max...')
    min_mz = min([np.amin(e) for e in listMz])
    max_mz = max([np.amax(e) for e in listMz])
    min_rt = listRt[0]
    max_rt = listRt[len(listRt)-1]

    # LOOK HERE adjust width and height here, bigger width and height yields higher resolution
    # width and height must be an int
    WIDTH = int(10*(max_mz - min_mz))
    HEIGHT = int(len(listRt)/3)

    print('converting spectra to 2d img...')
    # initialize whole_img as a 2d array specified height and width
    img = np.zeros((HEIGHT, WIDTH))
    listMz = [((e-min_mz)/(max_mz-min_mz)*(WIDTH-1)).astype('int') for e in tqdm(listMz)]
    listRt = ((np.array(listRt)-min_rt)/(max_rt-min_rt)*(HEIGHT-1)).astype('int')
    for x_list, y, inten_list in zip(listMz, listRt, listIn):
        # print(img.shape, inten_list.shape)
        img[y, x_list] += inten_list

    min_intensity = np.amin(img)
    max_intensity = np.amax(img)
    print('intensities:', max_intensity, min_intensity)

    print('transforming intensities...')
    img = (img - min_intensity) / (max_intensity - min_intensity)
    img = np.tanh(3*img)*255
    img = img.astype('int')

    file_name = file_name.split('.')[0]
    # LOOK HERE output is saved in the numpy array format in the misc folder and in image format in the whole_img folder
    np.save('misc/'+file_name+'_misc', np.array([max_rt, min_rt, max_mz, min_mz]))
    print("image save status:" + str(cv2.imwrite('whole_img/' + file_name + ".png", img)))
    del img

'''import sys

from decoder import *
import numpy as np

import time
from tqdm import tqdm
import cv2
# import matplotlib.pyplot as plt
import os
for file_name in os.listdir('source'):
    file_path = 'source/'+file_name
    t = time.time()
    print(f'starting to process file {file_path}')

    # decode the xml file and store the information in 'result', a list of 'spectrum' objects
    result = decode(file_path)

    # setting width and height, this can be any number we set

    print('initializing min and max...')
    min_mz = min([np.amin(spectrum.mzArr) for spectrum in result])
    max_mz = max([np.amax(spectrum.mzArr) for spectrum in result])
    min_rt = min([spectrum.retentionTime for spectrum in result])
    max_rt = max([spectrum.retentionTime for spectrum in result])

    WIDTH = 5*int(max_mz - min_mz)
    HEIGHT = len(result)

    print('converting spectra list to 2d array...')
    # initialize whole_img as a 2d array specified height and width
    img = np.zeros((HEIGHT, WIDTH))
    # iterate through the list of spectra
    for spectrum in tqdm(result):
        rt = spectrum.retentionTime
        # the y_index of all the peaks in this spectrum
        y_index = int((rt - min_rt)/(max_rt - min_rt)*(HEIGHT-1)+0.5)
        for mz, intensity in zip(spectrum.mzArr, spectrum.intenArr):
            # the x_index of the current peak
            x_index = int((mz-min_mz)/(max_mz-min_mz)*(WIDTH-1)+0.5)
            img[y_index][x_index] += intensity

    min_intensity = min([np.amin(spectrum) for spectrum in img])
    max_intensity = max([np.amax(spectrum) for spectrum in img])
    max_intensity /= 10

    print('normalizing intensities...')
    # eheheheheheheheeheheh
    for i in tqdm(range(len(img))):
        for j in range(len(img[i])):
            img[i, j] = int((img[i, j] - min_intensity)/(max_intensity - min_intensity)*255)
            img[i, j] = min(img[i, j], 255)
    file_name = file_name.split('.')[0]
    np.save('misc/'+file_name, img)
    np.save('misc/'+file_name+'_misc', np.array([max_rt, min_rt, max_mz, min_mz]))
    print("image status:" + str(cv2.imwrite('whole_img' + file_name + ".png", img)))
'''