import os, re
import json
from PIL import Image, ImageDraw
import PIL.ImageFile
import numpy as np
import scipy.ndimage

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted img_dataset/'
data = open('/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/improved_facial_landmarks img_dataset_v1/facial_landmarks.json')
data = json.load(data)
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/aligned img_dataset'

def recreate_aligned_images(json_data=data, dst_dir=dst,
							output_size=1024, transform_size=4096, enable_padding=True):
	if not os.path.exists(dst_dir):
		os.mkdir(dst_dir)


	for idx, item in enumerate(json_data['ontheradar'].values()):
		#print(item['filename'])

		# Parse landmarks
		lm = np.array(item['face_1_landmarks'])
		lm_chin = lm[0 : 17] # left-right
		lm_eyebrow_left = lm[17 : 22] # left-right
		lm_eyebrow_right = lm[22 : 27] # left-right
		lm_nose = lm[27 : 31] # top-down
		lm_nostrils = lm[31 : 36] # top-down
		lm_eye_left = lm[36 : 42] # left-clockwise
		lm_eye_right = lm[42 : 48] # left-clockwise
		lm_mouth_outer = lm[48 : 60] # left-clockwise
		lm_mouth_inner = lm[60 : 68] #left-clockwise

		# Calculate auxiliary vectors
		eye_left = np.mean(lm_eye_left, axis=0)
		eye_right = np.mean(lm_eye_right, axis=0)
		eye_avg = (eye_left + eye_right) * 0.5
		eye_to_eye = eye_right - eye_left
		mouth_left = lm_mouth_outer[0]
		mouth_right = lm_mouth_outer[6]
		mouth_avg = (mouth_left + mouth_right) * 0.5
		eye_to_mouth = mouth_avg - eye_avg

		# Choose oriented crop rectangle
		x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1] # ??
		x /= np.hypot(*x) # normalize
		x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8) # ??
		y = np.flipud(x) * [-1, 1]
		c = eye_avg + eye_to_mouth * 0.1 # (slightly) displaces eye_avg down the img frame: kind of defines the center of the face

		quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y]) # quadrilateral: pre-defines crop area
		q_wdth = quad[3, 0] - quad[0, 0]
		q_hth = quad[1, 1] - quad[0, 1]
		#quad[0, 0] -= q_wdth
		#quad[0, 3] += q_wdth
		#quad[]

		qsize = np.hypot(*x) * 2 # ??

		# Load in-the-wild image

		src_file = re.sub('/content/drive/My Drive/PIFuHD/SCULPTURES/', src, item['file_path'])

		if not os.path.isfile(src_file):
			print(f'Cannot find source image: {src_file}')
			return
		img = Image.open(src_file)

		draw = ImageDraw.Draw(img)
		draw.ellipse((eye_right[0] - 2, eye_right[1] - 2, eye_right[0] + 2, eye_right[1] + 2), fill=(255, 0, 0, 0))
		draw.ellipse((eye_left[0] - 2, eye_left[1] - 2, eye_left[0] + 2, eye_left[1] + 2), fill=(255, 0, 0, 0))
		draw.ellipse((eye_avg[0] - 2, eye_avg[1] - 2 , eye_avg[0] + 2, eye_avg[1] + 2), fill=(255, 0, 0, 0))
		#img.show()


		# Shrink
		#shrink = int(np.floor(qsize / output_size * 0.5))
		#if shrink > 1:
		#	print('/'.join(x for x in src_file.split('/')[-4:-2]) + '/' + src_file.split('/')[-1])
		#	rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
		#	img = img.resize(size=rsize, resample=Image.ANTIALIAS)
		#	quad /= shrink
		#	qsize /= shrink
		#img.show()


		# Crop
		border = max(int(np.rint(qsize * 0.1)), 3)
		crop = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
		crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))

		#crop = (int(np.floor(min(quad[:, 0]) - q_wdth)), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]) + q_wdth)), int(np.ceil(max(quad[:, 1]) + (6 * q_hth))))
		#crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))

		if crop[2] - crop [0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
			img = img.crop(crop)
			quad -= crop[0:2] # redefines quad within new (cropped) img's coordinates
		#img.show()


		# Pad
		pad = (int(np.floor(min(quad[:, 0]))), int(np.floor(min(quad[:, 1]))), int(np.ceil(max(quad[:, 0]))), int(np.ceil(max(quad[:, 1]))))
		pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
		if enable_padding and max(pad) > border - 4:
			img.show()
			print('/'.join(x for x in src_file.split('/')[-4:-2]) + '/' + src_file.split('/')[-1])
			pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
			img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
			h, w, _ = img.shape
			y, x, _ = np.ogrid[:h, :w, :1]
			mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
			blur = qsize * 0.02
			img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
			img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
			img = Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
			quad += pad[:2]
			img.show()


		# Transform
		img = img.transform(size=(transform_size, transform_size), method=Image.QUAD, data=(quad + 0.5).flatten(), resample=Image.BILINEAR) # ??
		img = img.transform(size=(transform_size, transform_size), method=Image.QUAD, data=(quad + 0.5).flatten(),
						   resample=Image.BILINEAR)
		if output_size < transform_size:
			img = img.resize((output_size, output_size), Image.ANTIALIAS)
		img.show()


		"""# Save aligned image
		folder = item['folder']
		rep = item['rep']
		filename = item['filename']
		dst_subdir = f'{dst_dir}/{folder}/{rep}'

		if not os.path.exists(dst_subdir):
			try:
				os.mkdir(dst_subdir)
			except FileNotFoundError:
				os.mkdir('/'.join(dst_subdir.split('/')[0:-1]))
				os.mkdir(dst_subdir)
		img.save(os.path.join(dst_subdir, filename))"""

recreate_aligned_images()