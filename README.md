# MS-feature
A deep learning-based feature detection tool for feature detection in Liquid Chromotography-Mass Spectrometry (LC-MS). Details available in [Deep Learning Based MS2 Feature Detection for Data-Independent Shotgun Proteomics](https://ieeexplore.ieee.org/abstract/document/9995258/metrics). The folder for Faster-RCNN framework is forked from [here](https://github.com/harshatejas/pytorch_custom_object_detection).

## Dependencies
- python==3.8
- torch==1.11.0
- torchvision==0.12.0
- opencv-python==4.6.0.66
- jupyter==1.0.0
- numpy==1.22.4
- tqdm==4.64.0

## Run
#### 
- Put source .mzml files into the ```data_prep/src``` folder
- Run ```mzml_to_img.ipynb``` jupyter notebook
- Adjust the height and width parameters in the first two lines ```generate_windows.ipynb``` for the sizes of the sliding windows. The default of 240x270 has been applied as stated in the paper.
- Run the first 4 cells of ```generate_windows.ipynb```. You should expect the windows in the folder named ```img_(height)_(width)```. 
- Execute ```predict.py```, changing line 11 to the folder generated from the last step.

