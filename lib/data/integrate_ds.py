import os
import shutil

#src = '/Users/ckxz/Downloads'
src = '/Users/ckxz/Desktop/zips'
dst = '/Volumes/CKXZ 1/@City/363, FP/AISculpture/PIFuHD/DS-Related/preprocessd_data/output'

current_folder = '7'

folders = [x for x in os.listdir(src) if x.startswith(current_folder)]

for folder in folders:
	print(folder)
	reps = [x for x in os.listdir(os.path.join(src, folder)) if not x.startswith('.')]
	for rep in reps:
		data_folders = [x for x in os.listdir(os.path.join(src, folder, rep)) if not x.startswith('.') and not x.endswith('.txt')]
		if not os.path.exists(os.path.join(dst, current_folder, rep)):
			os.mkdir((os.path.join(dst, current_folder, rep)))
		for dfolder in data_folders:
			if not os.path.exists(os.path.join(dst, current_folder, rep, dfolder)):
				os.mkdir((os.path.join(dst, current_folder, rep, dfolder)))
			if dfolder == 'GEO':
				if not os.path.exists(os.path.join(dst, current_folder, rep, dfolder, 'OBJ')):
					os.mkdir((os.path.join(dst, current_folder, rep, dfolder, 'OBJ')))
				#sculpts = [x for x in os.listdir(os.path.join(src, folder, rep, dfolder, 'OBJ')) if not x.startswith('.')]
				#for sculpt in sculpts:
				#	shutil.copy(os.path.join(src, folder, rep, dfolder, 'OBJ', sculpt), os.path.join(dst, current_folder, rep, dfolder, 'OBJ', sculpt))
				#	files = [x for x in os.listdir(os.path.join(src, folder, rep, dfolder, '0BJ', sculpt)) if not x.startswith('.')]
				#	for file in files:
				#		shutil.copy(os.path.join(src, folder, rep, dfolder, 'OBJ', sculpt, file), os.path.join(dst, current_folder, rep, dfolder, 'OBJ', sculpt, file))
				continue
			else:
				sculpts = [x for x in os.listdir(os.path.join(src, folder, rep, dfolder)) if
							  not x.startswith('.')]
				for sculpt in sculpts:
					if not os.path.exists(os.path.join(dst, current_folder, rep, dfolder, sculpt)):
						os.mkdir((os.path.join(dst, current_folder, rep, dfolder, sculpt)))

					files = [x for x in os.listdir(os.path.join(src, folder, rep, dfolder, sculpt)) if not x.startswith('.')]
					for file in files:
						shutil.copy(os.path.join(src, folder, rep, dfolder, sculpt, file), os.path.join(dst, current_folder, rep, dfolder, sculpt, file))
