import os
import bpy
import bmesh
import mathutils
from pathlib import Path

src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_reoriented_obj'
#src = '/Users/ckxz/Desktop/manifold_watertight' # meshes to check

folder_names = sorted([x for x in os.listdir(src) if not x.startswith('.') and not x.endswith('.txt')], key=int)
processed = []
watertight = []
non_watertight = []


# original from: https://blender.stackexchange.com/questions/160055/is-there-a-way-to-use-the-terminal-to-check-if-a-mesh-is-watertight
def is_watertight(object: bpy.types.Object, check_self_intersection=True) -> bool:
	"""
    Checks whether the given object is watertight or not
    :param object: Object the inspect
    :return: True if watertight, False otherwise
    """
	old_active_object = object
	old_mode = old_active_object.mode

	bpy.context.view_layer.objects.active = object

	bpy.ops.object.mode_set(mode='EDIT')
	bpy.ops.mesh.select_non_manifold(extend=False)
	bm = bmesh.from_edit_mesh(object.data)

	is_watertight = True

	for v in bm.verts:
		if v.select:
			is_watertight = False
			break

	# Only check for self intersection if manifold
	if is_watertight and check_self_intersection:
		bvh_tree = mathutils.bvhtree.BVHTree.FromBMesh(bm, epsilon=0.000001)
		intersections = bvh_tree.overlap(bvh_tree)

		if intersections:
			is_watertight = False

	bpy.context.view_layer.objects.active = old_active_object
	bpy.ops.object.mode_set(mode=old_mode)

	return is_watertight


for folder in folder_names:
	# reps = [x for x in os.listdir(f'{src}/{folder}') if not x.startswith('.')]
	# for rep in reps:
	rep = 'Statue'  # 'Bust'
	files = [x for x in os.listdir(os.path.join(src, folder, rep)) if not x.startswith('.') and not x.endswith('.mtl')]
	for file in files:
		if files in processed:
			continue
		else:
			bpy.ops.object.select_all(action='SELECT')
			bpy.ops.object.delete()
			bpy.ops.import_scene.obj(filepath=os.path.join(src, folder, rep, file), axis_forward='X', axis_up='Z')
			object_ = bpy.data.objects[0]
			if is_watertight(object_):
				processed.append(f'{folder}/{rep}/{file}')
				watertight.append(f'{folder}/{rep}/{file}')
			else:
				non_watertight.append(f'{folder}/{rep}/{file}')
				processed.append(f'{folder}/{rep}/{file}')

with open(f'{src}/non-watertight_STATUE.txt', 'w') as f:
	f.write(f'Nº of non-watertight meshes: {len(non_watertight)}, out of {len(processed)}\n')
	for file in non_watertight:
		f.write(f'{file}\n')

with open(f'{src}/watertight_STATUE.txt', 'w') as f:
	f.write(f'Nº of watertight meshes: {len(watertight)}, out of {len(processed)}\n')
	for file in watertight:
		f.write(f'{file}\n')