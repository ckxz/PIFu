import os
import bpy


src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/sorted stl_dataset_v3'
dst = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/shot_closeup_img_dataset'


folder_names = sorted([x for x in os.listdir(src) if not x.startswith('.')], key=int)

context = bpy.context.copy()


for area in bpy.context.screen.areas:
	if area.type == 'VIEW_3D':
		context['area'] = area
		bpy.ops.screen.screen_full_area(context, use_hide_panels=True)
		bpy.context.space_data.show_gizmo = False
		bpy.context.space_data.overlay.show_overlays = False


for folder in folder_names:
	if not os.path.exists(f'{dst}/{folder}'):
		os.mkdir(f'{dst}/{folder}')
	reps = [x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')]
	for rep in reps:
		if not os.path.exists(f'{dst}/{folder}/{rep}'):
			os.mkdir(f'{dst}/{folder}/{rep}')
		existent_files = [x for x in os.listdir(f'{dst}/{folder}/{rep}') if not x.startswith('.')]
		files = [x for x in os.listdir(f'{src}/{folder}/{rep}') if not x.startswith('.') and f'{x[:-4]}.png' not in existent_files]
		for file in files:
			bpy.ops.object.select_all(action='SELECT')
			bpy.ops.object.delete()
			bpy.ops.import_mesh.stl(filepath=f'{src}/{folder}/{rep}/{file}', axis_forward='-X', axis_up='Z')

			for area in bpy.context.screen.areas:
				if area.type == 'VIEW_3D':
					#context = bpy.context.copy()
					#context['area'] = area
					#bpy.ops.screen.screen_full_area(context, use_hide_panels=True)
					#bpy.context.space_data.show_gizmo = False
					#bpy.context.space_data.overlay.show_overlays = False
					for region in area.regions:
						if region.type == 'WINDOW':
							override = {'area': area, 'region': region, 'edit_object': bpy.context.edit_object}
							bpy.ops.view3d.view_all(override)
							bpy.ops.view3d.view_axis(override)
			bpy.ops.screen.screenshot(filepath=f'{dst}/{folder}/{rep}/{file[:-4]}.png')


