import os
from PIL import Image

#imagefile = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/shot_closeup_img_dataset/0/Statue/"Xanten Youth".png'
src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/shot_closeup_img_dataset'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/cropped_closeup_img_dataset'
folder_names = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)

for folder in folder_names:
	if not os.path.exists(f'{dst}/{folder}'):
		os.mkdir(f'{dst}/{folder}')
	reps = sorted([x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')])
	for rep in reps:
		if not os.path.exists(f'{dst}/{folder}/{rep}'):
			os.mkdir(f'{dst}/{folder}/{rep}')
		existent_files = [x for x in os.listdir(f'{dst}/{folder}/{rep}') if not x.startswith('.')]
		files = [x for x in os.listdir(f'{src}/{folder}/{rep}') if not x.startswith('.') and f'{x[:-4]}.png' not in existent_files]

		for file in files:
			image = Image.open(f'{src}/{folder}/{rep}/{file}')

			if rep == 'Statue':
				width, height = image.size
				left = (width / 2) - (width / 8)
				right = (width / 2) + (width / 8)
				top = height / 8
				bottom = (7 * height) / 8

				image = image.crop((left, top, right, bottom))
				#image.show()
				image.save(f'{dst}/{folder}/{rep}/{file}')

			elif rep == 'Bust':
				width, height = image.size
				left = (width / 2) - 1.25 * (width / 8)
				right = (width / 2) + 1.25 * (width / 8)
				top = height / 8
				bottom = (7 * height) / 8

				image = image.crop((left, top, right, bottom))
				#image.show()
				image.save(f'{dst}/{folder}/{rep}/{file}')

