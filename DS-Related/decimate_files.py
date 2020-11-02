import os
import bpy
from pathlib import Path

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/reoriented_obj_v3'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_obj'
#path2wtight_busts = os.path.join(dst, 'watertight_BUSTS.txt')
#path2wtight_statues = os.path.join(dst, 'watertight_STATUES.txt')

#watertight_bust = [x.strip('\n').split('/')[-1] for x in open(path2wtight_busts, 'r').readlines()][1:]
#watertight_statue = [x.strip('\n').split('/')[-1] for x in open(path2wtight_statues, 'r').readlines()][1:]
folders = sorted([x for x in os.listdir(src) if not x.startswith('.') and not x.endswith('.txt')], key=int)

median_size = 10


# Decimation script partly inspired in https://blender.stackexchange.com/questions/79924/how-can-i-apply-decimate-modifier-to-multple-files-and-save-them-as-a-different
def clean_modifiers(object):
	for mod in object.modifiers:
		if (mod.type == 'DECIMATE'):
			print('Removing modifier')
			object.modifiers.remove(modifier=mod)

for folder in folders:
	if not os.path.exists(os.path.join(dst, folder)):
		os.mkdir(os.path.join(dst, folder))
	reps = [x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')]
	for rep in reps:
	#rep = 'Bust' #'Bust'
		if not os.path.exists(os.path.join(dst, folder, rep)):
			os.mkdir(os.path.join(dst, folder, rep))
		files = [x for x in os.listdir(os.path.join(src, folder, rep)) if not x.startswith('.') and not x.endswith('.mtl')]
		for file in files:
			if os.path.exists(os.path.join(dst, folder, rep, file)):
				continue
			else:
				bpy.ops.object.select_all(action='SELECT')
				bpy.ops.object.delete()
				bpy.ops.import_scene.obj(filepath=os.path.join(src, folder, rep, file), axis_forward='X', axis_up='Z')
				filesize = (Path(os.path.join(src, folder, rep, file)).stat().st_size) / 10**6
				if filesize > median_size:
					#initial_facecount = len(bpy.context.object.data.polygons)
					#print(f'Initial face count: {initial_facecount}')
					decimate_ratio = median_size / filesize
					modifier_name = 'Decimate'
					# Select object to modify
					object = bpy.data.objects[0]
					# Remove any previously used modifiers
					clean_modifiers(object)
					# Apply modification
					modifier = object.modifiers.new(modifier_name, 'DECIMATE')
					modifier.ratio = decimate_ratio
					modifier.use_collapse_triangulate = True
				# Export
				bpy.ops.object.select_all(action='SELECT')
				bpy.ops.export_scene.obj(filepath=os.path.join(dst, folder, rep, file), axis_forward='X', axis_up='Z', use_materials=True)