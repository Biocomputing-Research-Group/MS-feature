import torch
import numpy as np
import cv2
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import time
import pandas as pd
from tqdm import tqdm
saved_model = "saved_model/ep_15"  # directory of the saved model
img_dir = 'dataset/validation/'  # input directory to conduct the inference on

NMS_THRESH = 0.12
SCORE_THRESH = 0.0

state = torch.load(saved_model, map_location=torch.device('cpu'))
with open('labels.txt', 'r') as f:
	string = f.read()
	labels_dict = eval(string)


def soft_nms_pytorch(dets, box_scores, sigma=0.5, thresh=0.0005, cuda=0):
	"""
	Build a pytorch implement of Soft NMS algorithm.
	# Augments
		dets:        boxes coordinate tensor (format:[y1, x1, y2, x2])
		box_scores:  box score tensors
		sigma:       variance of Gaussian function
		thresh:      score thresh
		cuda:        CUDA flag
	# Return
		the index of the selected boxes
	"""
	# Indexes concatenate boxes with the last column
	N = dets.shape[0]
	if cuda:
		indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
	else:
		indexes = torch.arange(0, N, dtype=torch.float).view(N, 1)
	dets = torch.cat((dets, indexes), dim=1)

	# The order of boxes coordinate is [y1,x1,y2,x2]
	y1 = dets[:, 0]
	x1 = dets[:, 1]
	y2 = dets[:, 2]
	x2 = dets[:, 3]
	scores = box_scores
	areas = (x2 - x1 + 1) * (y2 - y1 + 1)

	for i in range(N):
		# intermediate parameters for later parameters exchange
		tscore = scores[i].clone()
		pos = i + 1

		if i != N - 1:
			maxscore, maxpos = torch.max(scores[pos:], dim=0)
			if tscore < maxscore:
				dets[i], dets[maxpos.item() + i + 1] = dets[maxpos.item() + i + 1].clone(), dets[i].clone()
				scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
				areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

		# IoU calculate
		yy1 = np.maximum(dets[i, 0].to("cpu").numpy(), dets[pos:, 0].to("cpu").numpy())
		xx1 = np.maximum(dets[i, 1].to("cpu").numpy(), dets[pos:, 1].to("cpu").numpy())
		yy2 = np.minimum(dets[i, 2].to("cpu").numpy(), dets[pos:, 2].to("cpu").numpy())
		xx2 = np.minimum(dets[i, 3].to("cpu").numpy(), dets[pos:, 3].to("cpu").numpy())

		w = np.maximum(0.0, xx2 - xx1 + 1)
		h = np.maximum(0.0, yy2 - yy1 + 1)
		inter = torch.tensor(w * h).cuda() if cuda else torch.tensor(w * h)
		ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

		# Gaussian decay
		weight = torch.exp(-(ovr * ovr) / sigma)
		scores[pos:] = weight * scores[pos:]

	# select the boxes and keep the corresponding indexes
	keep = dets[:, 4][scores > thresh].int()

	return keep


def get_model(num_classes):
	# Load an pre-trained object detectin model (in this case faster-rcnn)
	model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

	# Number of input features
	in_features = model.roi_heads.box_predictor.cls_score.in_features

	# Replace the pre-trained head with a new head
	model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

	return model


loaded_model = get_model(len(labels_dict))
print(len(labels_dict))
# loaded_model.load_state_dict(torch.load(os.path.join(saved_model, 'model'), map_location='cpu'))
loaded_model.load_state_dict(state['state_dict'])
loaded_model.eval()

image_reading_time = 0.0
prediction_time = 0.0
nms_time = 0.0
n_predictions = []
n_labels = []
file_list = []
df = pd.read_csv('dataset/validation.csv')
output_csv = []

for filename in tqdm([e for e in os.listdir(img_dir) if 'pred' not in e]):  # Image filename
	print('processing ' + filename)
	img_path = img_dir + filename

	t = time.time()
	image = cv2.imread(img_path)
	img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	img = torchvision.transforms.ToTensor()(img)
	img_t = time.time() - t
	print(f"image reading time: {img_t}")
	image_reading_time += img_t

	t = time.time()
	with torch.no_grad():
		prediction = loaded_model([img])
	pred_t = time.time() - t
	print(f'prediction time: {pred_t}')
	prediction_time += pred_t
	t = time.time()
	keep = soft_nms_pytorch(prediction[0]['boxes'], prediction[0]['scores'], thresh=NMS_THRESH)
	nms_t = time.time() - t
	print(f"nms time: {nms_t}")
	nms_time += nms_t

	n_predictions.append(len(keep))
	n_labels.append(df['filename'].value_counts()[filename])
	file_list.append(filename)

	orig = image.copy()
	for element in keep:
		x, y, w, h = prediction[0]['boxes'][element].numpy().astype(int)
		score = np.round(prediction[0]['scores'][element].numpy(), decimals=3)
		label_index = prediction[0]['labels'][element].numpy()
		label = labels_dict[int(label_index)]
		if score < SCORE_THRESH:
			continue

		output_csv.append([filename, score, min(x, w), min(y, h), max(x, w), max(y, h)])
	'''
		cv2.rectangle(image, (x, y), (w, h), (70, 70, 255), 1)
		text = str(label) + " " + str(score)
		#cv2.putText(image, text, (x-40, h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
	print('image status: ' + cv2.imwrite(img_path.split('.')[0] + '-pred.jpg', image))
	for element in [e for _, e in df.iterrows() if filename == e['filename']]:
		x, y, w, h = element['xmin'], element['ymin'], element['xmax'], element['ymax']
		image = cv2.rectangle(image, (x, y), (w, h), (70, 255, 70), 1)
	print('image status: ' + str(cv2.imwrite(img_path.split('.')[0] + '-pred.jpg', image)))

	image = np.hstack((orig, image))
	cv2.imshow(filename, image)
	cv2.waitKey(0)
	'''

pd.DataFrame(output_csv, columns=['filename', 'score', 'xmin', 'ymin', 'xmax', 'ymax']).to_csv('dataset/predictions.csv')
print('total number of predictions: ' + str(sum(n_predictions)))
print(n_predictions)
print('total number of labels: ' + str(sum(n_labels)))
print(n_labels)
pd.DataFrame({'filename': file_list, 'predictions': n_predictions, 'maxquant': n_labels}).to_csv('detection_comparison.csv')
print('total prediction time: ' + str(prediction_time))
print('total non-max suppression time: ' + str(nms_time))
