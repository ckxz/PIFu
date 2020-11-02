import warnings
warnings.filterwarnings("ignore")

import os
import re
import json
import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw

# This script is partly inspired on https://stackoverflow.com/questions/30508079/python-center-array-image

gdrive_src = '/content/drive/My Drive/PIFuHD/SCULPTURES/'
local_src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted img_dataset/'

# Load facial landmarks
data = open('/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/improved_facial_landmarks img_dataset_v1/facial_landmarks.json')
data = json.load(data)

dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/aligned img_dataset'

# Get device right
device = torch.device('cpu')
if torch.cuda.is_available():
   device = torch.device('cuda')


# Download and initialize maskrcnn_resnet50 model
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, num_classes=91)
maskrcnn.eval()
maskrcnn = maskrcnn.to(device)

to_tensor = torchvision.transforms.ToTensor()


def process_image(img, landmarks, dst, folder, rep, filename, output_size=1024):
	"""Return a centered image.
	:param img: image as np.array
	:param landmarks: coordinates of facial features
	:param output_size: resolution of output image
	"""

	# Parse landmarks
	lm_chin = landmarks[0: 17] # left-right
	lm_eyebrow_left = landmarks[17: 22]  # left-right
	lm_eyebrow_right = landmarks[22: 27]  # left-right
	lm_nose = landmarks[27: 31]  # top-down
	lm_eye_left = landmarks[36: 42]  # left-clockwise
	lm_eye_right = landmarks[42: 48]  # left-clockwise
	lm_mouth_outer = landmarks[48: 60]  # left-clockwise

	# Calculate auxiliary vectors
	eye_left = np.mean(lm_eye_left, axis=0)
	eye_right = np.mean(lm_eye_right, axis=0)
	eye_avg = (eye_left + eye_right) * 0.5
	mouth_dot = np.mean(lm_mouth_outer, axis=0)
	nose_dot = np.mean(lm_nose, axis=0)
	center = np.rint(np.mean(np.array((eye_avg, mouth_dot, nose_dot)), axis=0))
	head_length = max(np.array(lm_chin)[:, 1]) - min(np.hstack((np.array(lm_eyebrow_left)[:, 1], np.array(lm_eyebrow_right)[:, 1])))
	head_width = max(np.array(lm_chin)[:, 0]) - min(np.array(lm_chin)[:, 0])

	y1 = int(np.rint(center[1] - (2 *head_length)))
	y2 = int(np.rint(center[1] + (13 * head_length)))
	x1 = int(np.rint(center[0]) - (3 * head_width))
	x2 = int(np.rint(center[0] + (3 * head_width)))



	t_img = to_tensor(img).to(device)

	output = maskrcnn([t_img])
	#scores = output[0]['scores'].detach().cpu()
	#bboxes = output[0]['boxes'].detach().cpu()
	#classes = output[0]['labels'].detach().cpu()
	masks = output[0]['masks'].detach().cpu()

	if masks.shape[0] == 1:
		masks_ = masks.squeeze().numpy()
	else:
		masks_ = torch.sum(masks.squeeze(), dim=0).numpy()

	mask = np.where(masks_ != 0)
	_x1_, _y1_, _x2_, _y2_ = min(mask[:][1]), min(mask[:][0]), max(mask[:][1]), max(mask[:][0])
	border_x = int(np.rint((_x2_ - _x1_) * 0.8))
	border_y = int(np.rint((_y2_ - _y1_) * 0.4))

	#img[mask] = 255
	#pil_img = Image.fromarray(np.uint8(img))
	#draw = ImageDraw.Draw(pil_img)
	#draw.ellipse((mouth_dot[0] - 2, mouth_dot[1] - 2, mouth_dot[0] + 2, mouth_dot[1] + 2), fill=(255))
	#draw.ellipse((nose_dot[0] - 2, nose_dot[1] - 2, nose_dot[0] + 2, nose_dot[1] + 2), fill=(255))
	#draw.ellipse((eye_avg[0] - 2, eye_avg[1] - 2, eye_avg[0] + 2, eye_avg[1] + 2), fill=(255))
	#draw.ellipse((center[0] - 2, eye_avg[1] - 2, eye_avg[0] + 2, eye_avg[1] + 2), fill=(0))
	#draw.rectangle([x1, y1, x2, y2], fill=None, outline=(255))
	#draw.rectangle([x1_, y1_, x2_, y2_], fill=None, outline=(0))
	#draw.rectangle([_x1_, _y1_, _x2_, _y2_], fill=None, outline=(255))
	#draw.rectangle([center[0] - head_width, center[1] + head_length, center[0] + head_width, center[1] - head_length], fill=None, outline=(255))
	#pil_img.show()

	_x1_ = max((_x1_ - border_x), 0)
	_x2_ = min((_x2_ + border_x), img.shape[0])
	#_y1_ = max((_y1_ - border_y), 0)
	if int(np.rint(center[1] - (output_size / 4))) < 0: # use border?
		_y1_ = 0
	else:
		_y1_ = int(np.rint(center[1]) - (output_size / 4))
	_y2_ = min((_y2_ + border_y), img.shape[1])
	cropped_image = img[_y1_:_y2_, _x1_:_x2_]
	Image.fromarray(np.uint8(cropped_image)).show()


	zero_axis_fill = (output_size - cropped_image.shape[1])
	one_axis_fill = (output_size - cropped_image.shape[0])

	left = int(np.rint(output_size / 2) - (center[0] - _x1_))
	right = zero_axis_fill - left
	#top = int(np.rint(one_axis_fill / 2))
	top = int(np.rint((output_size / 4) - (center[1] - _y1_)))
	bottom = one_axis_fill - top

	#pil_img_ = Image.fromarray(np.uint8(img))
	#draw_ = ImageDraw.Draw(pil_img_)
	#draw_.rectangle([left, top, right, bottom], fill=None, outline=(255))
	#pil_img_.show()

	try:
		padded_image = np.pad(cropped_image, ((top, bottom), (left, right)), mode='edge')
		img = Image.fromarray(np.uint8(padded_image)).convert('RGB')
		img.show()

		img.save(f'{dst}/{folder}/{rep}/{filename}.png')
		#return img
	except ValueError:
		print(f'ValueError: {folder}/{rep}/{filename}')
		print(f'center[1]: {center[1]}, _y1_: {_y1_}\n')
		print(f'center[0]: {center[0]}, _x1_: {_x1_}\n')
		unprocessed.append(f'{folder}/{rep}/{filename}')
		pass


for idx, item in enumerate(data['ontheradar'].values()):
		src_file = re.sub(gdrive_src, local_src, item['file_path'])
		folder = item['folder']
		rep = item['rep']
		filename = ''.join(x for x in item['filename'].split('front')[0])
		#print(filename)
		unprocessed = []

		if not os.path.exists(f'{dst}/{folder}/{rep}'):
			try:
				os.mkdir(f'{dst}/{folder}/{rep}')
			except FileNotFoundError:
				os.mkdir(f'{dst}/{folder}')
				os.mkdir(f'{dst}/{folder}/{rep}')

		#Image.open(src_file).convert('L').show()
		img = np.array(Image.open(src_file).convert('L'))
		process_image(img, item['face_1_landmarks'], dst=dst, folder=folder, rep=rep, filename=filename)