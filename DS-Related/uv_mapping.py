import os
import bpy

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_obj'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_vt'
#path2watertight_busts = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_obj/watertight_BUST.txt'
path2watertight_statues = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_obj/watertight_STATUE.txt'

#watertight_bust = [x.strip('\n').split('/')[-1] for x in open(path2watertight_busts, 'r').readlines()][1:]
watertight_statue = [x.strip('\n').split('/')[-1] for x in open(path2watertight_statues, 'r').readlines()][1:]
folders = sorted([x for x in os.listdir(src) if not x.startswith('.') and not x.endswith('.txt')], key=int)


for folder in folders:
	if not os.path.exists(os.path.join(dst, folder)):
		os.mkdir(os.path.join(dst, folder))
	reps = [x for x in os.listdir(os.path.join(src, folder)) if not x.startswith('.')]
	#for rep in reps:
	rep = 'Statue'
	if rep == 'Statue':
		if not os.path.exists(os.path.join(dst, folder, rep)):
			os.mkdir(os.path.join(dst, folder, rep))
		files = [x for x in os.listdir(os.path.join(src, folder, rep)) if not x.startswith('.') and not x.endswith(('.mtl', '.png'))]
		for file in files:
			if os.path.exists(os.path.join(dst, folder, rep, file)):
				continue

			else:
				#if file in watertight_bust or file in watertight_statue:
				if file in watertight_statue:
					bpy.ops.object.select_all(action='SELECT')
					bpy.ops.object.delete()

					bpy.ops.import_scene.obj(filepath=os.path.join(src, folder, rep, file), axis_forward='X', axis_up='Z')
					object_ = bpy.data.objects[0]

					old_active_object = object_
					old_mode = old_active_object.mode

					bpy.context.view_layer.objects.active = object_

					bpy.ops.object.editmode_toggle()
					bpy.ops.mesh.select_all(action='SELECT')
					bpy.ops.uv.smart_project()

					bpy.context.view_layer.objects.active = old_active_object
					bpy.ops.object.mode_set(mode=old_mode)

					bpy.ops.uv.export_layout(filepath=os.path.join(dst, folder, rep, file[:-4] + '.png'), size=(2048, 2048))
					bpy.ops.export_scene.obj(filepath=os.path.join(dst, folder, rep, file), axis_forward='X', axis_up='Z',
											 use_materials=True)

				else:
					continue