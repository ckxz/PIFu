import os, sys
import random
import argparse
from pathlib import Path
from tqdm import tqdm

import cv2
import math
import pyexr
import shutil
import numpy as np

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from lib.renderer.camera import Camera
from lib.renderer.mesh import load_obj_mesh, compute_tangent, compute_normal, load_obj_mesh_mtl
# from data.config import raw_dataset, render_dataset, archive_dataset, model_list, zip_path


def make_rotate(rx, ry, rz):
	sinX = np.sin(rx)
	sinY = np.sin(ry)
	sinZ = np.sin(rz)

	cosX = np.cos(rx)
	cosY = np.cos(ry)
	cosZ = np.cos(rz)

	Rx = np.zeros((3, 3))
	Rx[0, 0] = 1.0
	Rx[1, 1] = cosX
	Rx[1, 2] = -sinX
	Rx[2, 1] = sinX
	Rx[2, 2] = cosX

	Ry = np.zeros((3, 3))
	Ry[0, 0] = cosY
	Ry[0, 2] = sinY
	Ry[1, 1] = 1.0
	Ry[2, 0] = -sinY
	Ry[2, 2] = cosY

	Rz = np.zeros((3, 3))
	Rz[0, 0] = cosZ
	Rz[0, 1] = -sinZ
	Rz[1, 0] = sinZ
	Rz[1, 1] = cosZ
	Rz[2, 2] = 1.0

	R = np.matmul(np.matmul(Rz, Ry), Rx)
	return R


def rotateSH(SH, R):
	SHn = SH

	# 1st order
	SHn[1] = R[1, 1] * SH[1] - R[1, 2] * SH[2] + R[1, 0] * SH[3]
	SHn[2] = -R[2, 1] * SH[1] + R[2, 2] * SH[2] - R[2, 0] * SH[3]
	SHn[3] = R[0, 1] * SH[1] - R[0, 2] * SH[2] + R[0, 0] * SH[3]

	# 2nd order
	SHn[4:, 0] = rotateBand2(SH[4:, 0], R)
	SHn[4:, 1] = rotateBand2(SH[4:, 1], R)
	SHn[4:, 2] = rotateBand2(SH[4:, 2], R)

	return SHn


def rotateBand2(x, R):
	s_c3 = 0.94617469575
	s_c4 = -0.31539156525
	s_c5 = 0.54627421529

	s_c_scale = 1.0 / 0.91529123286551084
	s_c_scale_inv = 0.91529123286551084

	s_rc2 = 1.5853309190550713 * s_c_scale
	s_c4_div_c3 = s_c4 / s_c3
	s_c4_div_c3_x2 = (s_c4 / s_c3) * 2.0

	s_scale_dst2 = s_c3 * s_c_scale_inv
	s_scale_dst4 = s_c5 * s_c_scale_inv

	sh0 = x[3] + x[4] + x[4] - x[1]
	sh1 = x[0] + s_rc2 * x[2] + x[3] + x[4]
	sh2 = x[0]
	sh3 = -x[3]
	sh4 = -x[1]

	r2x = R[0][0] + R[0][1]
	r2y = R[1][0] + R[1][1]
	r2z = R[2][0] + R[2][1]

	r3x = R[0][0] + R[0][2]
	r3y = R[1][0] + R[1][2]
	r3z = R[2][0] + R[2][2]

	r4x = R[0][1] + R[0][2]
	r4y = R[1][1] + R[1][2]
	r4z = R[2][1] + R[2][2]

	sh0_x = sh0 * R[0][0]
	sh0_y = sh0 * R[1][0]
	d0 = sh0_x * R[1][0]
	d1 = sh0_y * R[2][0]
	d2 = sh0 * (R[2][0] * R[2][0] + s_c4_div_c3)
	d3 = sh0_x * R[2][0]
	d4 = sh0_x * R[0][0] - sh0_y * R[1][0]

	sh1_x = sh1 * R[0][2]
	sh1_y = sh1 * R[1][2]
	d0 += sh1_x * R[1][2]
	d1 += sh1_y * R[2][2]
	d2 += sh1 * (R[2][2] * R[2][2] + s_c4_div_c3)
	d3 += sh1_x * R[2][2]
	d4 += sh1_x * R[0][2] - sh1_y * R[1][2]

	sh2_x = sh2 * r2x
	sh2_y = sh2 * r2y
	d0 += sh2_x * r2y
	d1 += sh2_y * r2z
	d2 += sh2 * (r2z * r2z + s_c4_div_c3_x2)
	d3 += sh2_x * r2z
	d4 += sh2_x * r2x - sh2_y * r2y

	sh3_x = sh3 * r3x
	sh3_y = sh3 * r3y
	d0 += sh3_x * r3y
	d1 += sh3_y * r3z
	d2 += sh3 * (r3z * r3z + s_c4_div_c3_x2)
	d3 += sh3_x * r3z
	d4 += sh3_x * r3x - sh3_y * r3y

	sh4_x = sh4 * r4x
	sh4_y = sh4 * r4y
	d0 += sh4_x * r4y
	d1 += sh4_y * r4z
	d2 += sh4 * (r4z * r4z + s_c4_div_c3_x2)
	d3 += sh4_x * r4z
	d4 += sh4_x * r4x - sh4_y * r4y

	dst = x
	dst[0] = d0
	dst[1] = -d1
	dst[2] = d2 * s_scale_dst2
	dst[3] = -d3
	dst[4] = d4 * s_scale_dst4

	return dst


def render_prt_ortho(out_path, obj_uv_filespath, prep_filespath, shs, rndr, rndr_uv, im_size, angl_step=4, n_light=1,
					 pitch=[0]):
	geo_path = Path(os.path.join(out_path, 'GEO', 'OBJ', objnuv_filepath.split('/')[-1]))
	param_path = Path(os.path.join(out_path, 'PARAM', objnuv_filepath.split('/')[-1]))
	# print(param_path)
	render_path = Path(os.path.join(out_path, 'RENDER', objnuv_filepath.split('/')[-1]))
	# print(render_path)
	mask_path = Path(os.path.join(out_path, 'MASK', objnuv_filepath.split('/')[-1]))
	# print(mask_path)
	uv_render_path = Path(os.path.join(out_path, 'UV_RENDER', objnuv_filepath.split('/')[-1]))
	# print(uv_render_path)
	uv_mask_path = Path(os.path.join(out_path, 'UV_MASK', objnuv_filepath.split('/')[-1]))
	# print(uv_mask_path)
	uv_pos_path = Path(os.path.join(out_path, 'UV_POS', objnuv_filepath.split('/')[-1]))
	# print(uv_pos_path)
	uv_normal_path = Path(os.path.join(out_path, 'UV_NORMAL', objnuv_filepath.split('/')[-1]))
	# print(uv_normal_path)

	if os.path.exists(os.path.join(geo_path, objnuv_filepath.split('/')[-1] + '.obj')) and \
			os.path.exists(os.path.join(param_path, '359_0_00.npy')) and \
			os.path.exists(os.path.join(render_path, '359_0_00.jpg')) and \
			os.path.exists(os.path.join(mask_path, '359_0_00.png')) and \
			os.path.exists(os.path.join(uv_render_path, '359_0_00.jpg')) and \
			os.path.exists(os.path.join(uv_mask_path, '00.png')) and \
			os.path.exists(os.path.join(uv_pos_path, '00.exr')) and \
			os.path.exists(os.path.join(uv_normal_path, '00.png')):
		print('Files exist, stepping out.')
		return
	else:
		os.makedirs(geo_path, exist_ok=True)
		os.makedirs(param_path, exist_ok=True)
		os.makedirs(render_path, exist_ok=True)
		os.makedirs(mask_path, exist_ok=True)
		os.makedirs(uv_render_path, exist_ok=True)
		os.makedirs(uv_mask_path, exist_ok=True)
		os.makedirs(uv_pos_path, exist_ok=True)
		os.makedirs(uv_normal_path, exist_ok=True)

	cam = Camera(width=im_size, height=im_size)
	cam.ortho_ratio = 0.4 * (512 / im_size)
	cam.near = -100
	cam.far = 100
	cam.sanity_check()

	# set path for obj, prt
	mesh_file = obj_uv_filespath + '.obj'
	# mesh_file = '/content/drive/My Drive/untitled.obj'
	if not os.path.exists(mesh_file):
		print('ERROR: obj file does not exist!!', mesh_file)
		return
	shutil.copy(mesh_file, os.path.join(geo_path, mesh_file.split('/')[-1]))
	#with open ('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/PIFu/mesh.txt', 'w') as f:
	#	f.write('Mesh copied.')
	text_file = obj_uv_filespath + '.png'
	# text_file = '/content/drive/My Drive/PIFuHD/decimated_dataset/0/Bust/"The Younger Memnon", Colossal bust of Ramesses II.png'
	if not os.path.exists(text_file):
		print('ERROR: dif file does not exist!!', text_file)
		return

	prt_file = prep_filespath + 'bounce.txt'
	print(prt_file)
	# prt_file = '/content/drive/My Drive/PIFuHD/preprocessing/prt_util_decimated/0/Bust/"The Younger Memnon", Colossal bust of Ramesses II__bounce.txt'
	if not os.path.exists(prt_file):
		print('ERROR: prt file does not exist!!!', prt_file)
		return
	face_prt_file = prep_filespath + 'face.npy'
	# face_prt_file = '/content/drive/My Drive/PIFuHD/preprocessing/prt_util_decimated/0/Bust/"The Younger Memnon", Colossal bust of Ramesses II__face.npy'
	if not os.path.exists(face_prt_file):
		print('ERROR: face prt file does not exist!!!', prt_file)
		return

	texture_image = cv2.imread(text_file)
	texture_image = cv2.cvtColor(texture_image, cv2.COLOR_BGR2RGB)

	vertices, faces, normals, faces_normals, textures, face_textures = load_obj_mesh(mesh_file, with_normal=True,
																					 with_texture=True)
	print(
		f'vertices:  {vertices.shape}, faces: {faces.shape}, normals: {normals.shape}, face_normals: {faces_normals.shape}, textures: {textures.shape}, face_textures: {face_textures.shape}')

	# vertices, faces, normals, face_normals = load_obj_mesh(mesh_file, with_normal=True, with_texture=False)
	vmin = vertices.min(0)
	vmax = vertices.max(0)
	up_axis = 1 if (vmax - vmin).argmax() == 1 else 2

	vmed = np.median(vertices, 0)
	vmed[up_axis] = 0.5 * (vmax[up_axis] + vmin[up_axis])
	y_scale = 180 / (vmax[up_axis] - vmin[up_axis])

	rndr.set_norm_mat(y_scale, vmed)
	rndr_uv.set_norm_mat(y_scale, vmed)

	tan, bitan = compute_tangent(vertices, faces, normals, textures, face_textures)
	# tan, bitan = compute_tangent_(vertices, faces, normals)
	prt = np.loadtxt(prt_file)
	face_prt = np.load(face_prt_file)
	rndr.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
	rndr.set_albedo(texture_image)

	rndr_uv.set_mesh(vertices, faces, normals, faces_normals, textures, face_textures, prt, face_prt, tan, bitan)
	rndr_uv.set_albedo(texture_image)

	if not os.path.exists(os.path.join(out_path, 'val.txt')):
		f = open(os.path.join(out_path, 'val.txt'), 'w')
		f.close()

	# copy obj file
	cmd = 'cp %s %s' % (mesh_file, os.path.join(out_path, 'GEO', 'OBJ', objnuv_filepath.split('/')[-1]))
	print(cmd)
	os.system(cmd)

	for p in pitch:
		for y in tqdm(range(0, 360, angl_step)):
			R = np.matmul(make_rotate(math.radians(p), 0, 0), make_rotate(0, math.radians(y), 0))
			if up_axis == 2:
				R = np.matmul(R, make_rotate(math.radians(-90), 0, 0))

			rndr.rot_matrix = R
			rndr_uv.rot_matrix = R
			rndr.set_camera(cam)
			rndr_uv.set_camera(cam)

			for j in range(n_light):
				sh_id = random.randint(0, shs.shape[0] - 1)
				sh = shs[sh_id]
				sh_angle = 0.2 * np.pi * (random.random() - 0.5)
				sh = rotateSH(sh, make_rotate(0, sh_angle, 0).T)

				dic = {'sh': sh, 'ortho_ratio': cam.ortho_ratio, 'scale': y_scale, 'center': vmed, 'R': R}

				rndr.set_sh(sh)
				rndr.analytic = False
				rndr.use_inverse_depth = False
				rndr.display()

				out_all_f = rndr.get_color(0)
				out_mask = out_all_f[:, :, 3]
				out_all_f = cv2.cvtColor(out_all_f, cv2.COLOR_RGBA2BGR)

				np.save(os.path.join(param_path, '%d_%d_%02d.npy' % (y, p, j)), dic, allow_pickle=True)
				cv2.imwrite(os.path.join(render_path, '%d_%d_%02d.jpg' % (y, p, j)),
							255.0 * out_all_f)
				cv2.imwrite(os.path.join(mask_path, '%d_%d_%02d.png' % (y, p, j)),
							255.0 * out_mask)

				rndr_uv.set_sh(sh)
				rndr_uv.analytic = False
				rndr_uv.use_inverse_depth = False
				rndr_uv.display()

				uv_color = rndr_uv.get_color(0)
				uv_color = cv2.cvtColor(uv_color, cv2.COLOR_RGBA2BGR)
				cv2.imwrite(os.path.join(uv_render_path, '%d_%d_%02d.jpg' % (y, p, j)),
							255.0 * uv_color)

				if y == 0 and j == 0 and p == pitch[0]:
					uv_pos = rndr_uv.get_color(1)
					uv_mask = uv_pos[:, :, 3]
					cv2.imwrite(os.path.join(uv_mask_path, '00.png'), 255.0 * uv_mask)

					data = {'default': uv_pos[:, :, :3]}  # default is a reserved name
					pyexr.write(os.path.join(uv_pos_path, '00.exr'), data)

					uv_nml = rndr_uv.get_color(2)
					uv_nml = cv2.cvtColor(uv_nml, cv2.COLOR_RGBA2BGR)
					cv2.imwrite(os.path.join(uv_normal_path, '00.png'), 255.0 * uv_nml)





# RUN
#wtight_bust = [x[:-1] for x in open('/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_BUSTS.txt', 'r').readlines() if '.obj' in x] # Local
#wtight_statue = [x[:-1] for x in open('/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_STATUES.txt', 'r').readlines() if '.obj' in x] # Local

wtight_bust =  [x[:-1] for x in open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_reoriented_vt/watertight_BUST.txt', 'r').readlines() if '.obj' in x] # Camber
wtight_statue = [x[:-1] for x in open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_reoriented_vt/watertight_STATUE.txt', 'r').readlines() if '.obj' in x] # Camber

#file_src = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset_vt' # Local
file_src = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_reoriented_vt' # Camber
#prep_src = '/Volumes/CKXZ 1/@City/363, FP/AISculpture/PIFuHD/DS-Related/preprocessd_data/prt_util' # Local
prep_src = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/prt_util_reoriented' # Camber
#dst = '/Volumes/CKXZ 1/@City/363, FP/AISculpture/PIFuHD/DS-Related/preprocessd_data/output_tryitlocal' # Local
dst = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/pp_output_reoriented' # Camber
#env_sh = '/Users/ckxz/Desktop/@City/363, FP/PIFu/env_sh.npy' # Local
env_sh = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/PIFu/env_sh.npy'

folders = sorted([x for x in os.listdir(file_src) if not x.startswith('.') and not x.endswith(('.txt', '.zip'))], key=int)
shs = np.load(env_sh)

from lib.renderer.gl.init_gl import initialize_GL_context

initialize_GL_context(width=512, height=512, egl=True)

from lib.renderer.gl.prt_render import PRTRender

rndr = PRTRender(width=512, height=512, ms_rate=1, egl=True)
rndr_uv = PRTRender(width=512, height=512, uv_mode=True, egl=True)

#ccount = 0
#fcount = 0
#ftcount = 0

for folder in folders:
	#if not os.path.exists(os.path.join(dst, folder)):
	#	os.mkdir(os.path.join(dst, folder))
	reps = [x for x in os.listdir(f'{file_src}/{folder}') if not x.startswith('.')]
	for rep in reps:
		if not os.path.exists(os.path.join(dst, rep)):
			os.mkdir(os.path.join(dst, rep))
		files = [x for x in os.listdir(os.path.join(file_src, folder, rep)) if not x.startswith('.') and not x.endswith(('.mtl', '.png'))]
		for fname in files:
			if os.path.join(folder, rep, fname) not in wtight_bust and os.path.join(folder, rep, fname) not in wtight_statue:
				#ccount += 1
				#with open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/ccount.txt', 'w') as f:
				#	f.write(str(ccount))
				continue
			else:
				#fcount += 1
				#with open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/fcount.txt', 'w') as f:
				#	f.write(str(fcount))
				objnuv_filepath = os.path.join(file_src, folder, rep, fname[:-4])
				print(objnuv_filepath.split('/')[-1])
				prep_filespath = os.path.join(prep_src, folder, rep, fname[:-4] + '__')
				dst_path = os.path.join(dst, rep)
				render_prt_ortho(dst_path, objnuv_filepath, prep_filespath, shs, rndr, rndr_uv, 512, 1, 1, pitch=[0])
				#ftcount += 1
				#with open('/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/ftcount.txt', 'w') as f:
				#	f.write(str(ftcount))

