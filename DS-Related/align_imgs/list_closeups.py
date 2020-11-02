import os
import pandas as pd

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/cropped_closeup_img_dataset'

folder_names = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)

id_names = []
sculptures = []

for folder in folder_names:
	reps = [x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')]
	for rep in reps:
		files = [x for x in os.listdir(f'{src}/{folder}/{rep}') if not x.startswith('.')]
		for idx, file in enumerate(files):
			sculptures.append(file)
			id_names.append(f'{folder}/{str(rep[0]).lower()}{idx}')

sculptures_n = len(sculptures)
busts = len([x for x in id_names if 'b' in x])
statues = len([x for x in id_names if 's' in x])
ratio_b = (busts / sculptures_n) * 100
ratio_s = (statues / sculptures_n) * 100

print(f'Nº of busts: {busts} ({ratio_b}%).\n'
	  f'Nº of statues: {statues} ({ratio_s}%).\n'
	  f'Total nº of sculptures: {sculptures_n}.')

df = pd.DataFrame()
df['id'] = id_names
df['Name'] = sculptures
df.to_csv('/Users/ckxz/Desktop/@City/363, FP/AISculpture/PIFuHD/DS-Related/cropped_closeup_img_dataset_list.csv', index=False)

