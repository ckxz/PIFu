from collections import OrderedDict
from imutils import face_utils
import imutils
import json
import dlib
import cv2
import os

#os.environ['KMP_DUPLICATE_LIB_OK']='True' # OMP: Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized.

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted img_dataset'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/facial_landmarks img_dataset'
shape_predictor = '/Users/ckxz/Desktop/@City/363, FP/AISculpture/PIFuHD/DS-Related/shape_predictor_68_face_landmarks.dat'


# This script is inspired on https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/

# initialize dlib's face detector (HOG-based) and then create the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shape_predictor)

none_detected = []

data = OrderedDict()
data['ontheradar'] = OrderedDict()
data['undertheradar'] = OrderedDict()

folders = [x for x in os.listdir(src) if not x.startswith('.')]

n = 0
t = 0

on_img_idx = 1
under_img_idx = 1

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
					# load the input image, resize it, and convert it to grayscale
					img_file = f'{src}/{folder}/{rep}/{view}/{img}'
					image = cv2.imread(img_file)
					#image = imutils.resize(image, width=500)
					#cv2.imshow('Check', image)
					#cv2.waitKey(0)
					gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
					#cv2.imshow('Check', gray)
					#cv2.waitKey(0)

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
							json_shape = shape.tolist()
							data['ontheradar'][on_img_idx][f'face_{i + 1}_landmarks'] = shape.tolist()

							# convert dlib's rectangle to a OpenCV-style bounding box [i.e., (x, y, w, h)], then draw the face bounding box
							(x, y, w, h) = face_utils.rect_to_bb(rect)
							cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

							# show the face number
							cv2.putText(image, f'Face #{i + 1}', (x - 10, y - 10),
										cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

							# loop over the (x, y)-coordinates for the facial landmarks and draw them on the image
							for (x, y) in shape:
								cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

						on_img_idx += 1

						# save/show the output image with the face detections + facial landmarks
						img_name = img.split('front')[0] + '.png'
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
