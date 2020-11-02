import os
import shutil

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted obj_dataset_v3'
missing = [l.strip('\n') for l in open('/363, FP/AISculpture/PIFu/DS-Related/missing_files.txt', 'r', encoding='utf-8').readlines()]
print(len(missing))
dst = '/Users/ckxz/Desktop/missing_files'
#files_dataset = []

folders = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)[:23]
#folders_red = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)[:23]

#print((len(folders_red)/len(folders))*100)

for folder in folders:
	reps = [x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')]
	for rep in reps:
		files = [x for x in os.listdir(f'{src}/{folder}/{rep}') if not x.startswith('.')]
		for file in files:
			if file in missing:
				if not os.path.exists(f'{dst}/{folder}/{rep}'):
					os.makedirs(f'{dst}/{folder}/{rep}')
				shutil.copy(f'{src}/{folder}/{rep}/{file}', f'{dst}/{folder}/{rep}/{file}')

#with open(f'{dst}/orgnl_objv3_files.txt', 'w') as f:
#	for file in files_dataset:
#		f.write(f'{file}\n')