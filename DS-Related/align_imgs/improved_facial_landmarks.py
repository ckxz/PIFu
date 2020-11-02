import warnings
warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image as PILImage
import torchvision
from torchvision import transforms

from collections import OrderedDict
from imutils import face_utils

import json
import dlib
import cv2
import os

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted img_dataset'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/improved_facial_landmarks img_dataset_v1'
shape_predictor = '/Users/ckxz/Desktop/@City/363, FP/AISculpture/PIFuHD/DS-Related/shape_predictor_68_face_landmarks.dat'


# This script is partly inspired on https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/


# download and initialize maskrcnn_resnet50 model
maskrcnn = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True, num_classes=91)
maskrcnn.eval()

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

# initialize transforms
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage(mode='RGB')


# Define util data objects
none_detected = []

data = OrderedDict()
data['ontheradar'] = OrderedDict()
data['undertheradar'] = OrderedDict()

folders = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)[0]
print(folders)


n = 0
t = 0

on_img_idx = 1
under_img_idx = 1

# Run loop
for folder in folders:
	if not os.path.exists(f'{dst}/{folder}'):
		os.mkdir(f'{dst}/{folder}')
	reps = [x for x in os.listdir(os.path.join(src, folder)) if not x.startswith('.')]

	for rep in reps:
		if not os.path.exists(f'{dst}/{folder}/{rep}'):
			os.mkdir(f'{dst}/{folder}/{rep.lower().capitalize()}')
		views = [x for x in os.listdir(f'{src}/{folder}/{rep}') if not x.startswith('.')]

		for view in views:
			if 'front' in view.lower():
				imgs = [x for x in os.listdir(f'{src}/{folder}/{rep}/{view}') if
						 not x.startswith('.')]

				for img in imgs:
					img_name = img.split('front')[0] + '.png'

					if os.path.exists(f'{dst}/{folder}/{rep.lower().capitalize()}/{img_name}'):
						continue

					else:
						# load input image, get rid of alpha channel (if required) and pass it through maskrcnn
						img_file = f'{src}/{folder}/{rep}/{view}/{img}'

						image = np.array(PILImage.open(img_file))

						if image.shape[-1] != 3:
							image = image[:, :, :3]

						image = to_tensor(image)

						output = maskrcnn([image])

						scores = output[0]['scores'].cpu()
						bboxes = output[0]['boxes'].cpu()
						classes = output[0]['labels'].cpu()
						mask = output[0]['masks'].cpu()

						if len(scores) > 0:
							one_call_instance = np.where(max(scores.tolist()))
							bboxes = np.array(bboxes[one_call_instance].detach().squeeze())
							classes = classes[one_call_instance]
							mask = mask[one_call_instance]

							x1, y1, x2, y2 = bboxes[0], bboxes[1], bboxes[2], bboxes[3]

							image = image.cpu()
							image = to_pil(image)
							image = image.crop((x1, y1, x2, y2))
							image = np.array(image)[:, :, ::-1].copy()
							gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						else:
							image = image.cpu()
							image = to_pil(image)
							image = np.array(image)[:, :, ::-1].copy()
							gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

						# detect faces in the grayscale image
						rects = detector(gray, 1)

						if len(rects) == 0:
							data['undertheradar'][under_img_idx] = {
								'folder': folder,
								'rep': rep.lower().capitalize(),
								'filename': img,
								'file_path': img_file
							}
							under_img_idx += 1

							none_detected.append(f'{src}/{folder}/{rep}/{view}/{img}')
							n += 1
							continue

						else:
							# loop over the face detections
							data['ontheradar'][on_img_idx] = {
								'folder': folder,
								'rep': rep.lower().capitalize(),
								'filename': img,
								'file_path': img_file
							}
							for i, rect in enumerate(rects):
								# determine the facial landmarks for the face region, then convert the facial landmark (x, y)-coordinates to a NumPy array
								shape = predictor(gray, rect)
								shape = face_utils.shape_to_np(shape)
								json_shape = shape + np.array((round(x1), round(y1)))
								data['ontheradar'][on_img_idx][f'face_{i + 1}_landmarks'] = json_shape.tolist()

								# convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
								(x, y, w, h) = face_utils.rect_to_bb(rect)
								cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

								# show the face number
								cv2.putText(image, f'Face #{i + 1}', (x - 10, y - 10),
											cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

								# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
								for (x, y) in shape:
									cv2.circle(image, (x, y), 1, (0, 0, 255), -1)
								#cv2.imshow('Check', original)
								#cv2.waitKey(0)


							on_img_idx += 1

							# save/show the output image with the face detections + facial landmarks
							cv2.imwrite(f'{dst}/{folder}/{rep.lower().capitalize()}/{img_name}', image)
							t += 1
			else:
				pass



print(f'Imgs with no detected faces: {n}\n',
	  f'Images with detected faces: {t}\n',
	  f'Total: {n + t}\n',
	  f'% of imgs correctly processed: {(t / (t + n)) * 100}% out of {t+n}\n')


with open(f'{dst}/facial_landmarks.json', 'w') as file:
	json.dump(data, file)

with open(f'{dst}/noface_detected.txt', 'w') as file:
	for element in none_detected:
		file.write(f'{element}\n')