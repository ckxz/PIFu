import os
import math
import argparse
from tqdm import tqdm
from pathlib import Path

import trimesh
import numpy as np
from scipy.special import sph_harm


#watertight_busts_path = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_BUSTS.txt' # Local
#watertight_statues_path = '/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_STATUES.txt' # Local
root = '/home/enterprise.internal.city.ac.uk/adbb120/pifuhd' # Server
watertight_busts_path = os.path.join(root, 'data/decimated_obj-dataset_vt/watertight_BUSTS.txt') # Server
watertight_statues_path = os.path.join(root, 'data/decimated_obj-dataset_vt/watertight_STATUES.txt') # Server

watertight_bust = [x.strip('\n').split('/')[-1] for x in open(watertight_busts_path, 'r').readlines()][1:]
watertight_statue = [x.strip('\n').split('/')[-1] for x in open(watertight_statues_path, 'r').readlines()][1:]

def factratio(N, D):
	if N >= D:
		prod = 1.0
		for i in range(D + 1, N + 1):
			prod *= i
		return prod
	else:
		prod = 1.0
		for i in range(N + 1, D + 1):
			prod *= i
		return 1.0 / prod


def KVal(M, L):
	return math.sqrt(((2 * L + 1) / (4 * math.pi)) * (factratio(L - M, L + M)))


def AssociatedLegendre(M, L, x):
	if M < 0 or M > L or np.max(np.abs(x)) > 1.0:
		return np.zeros_like(x)

	pmm = np.ones_like(x)
	if M > 0:
		somx2 = np.sqrt((1.0 + x) * (1.0 - x))
		fact = 1.0
		for i in range(1, M + 1):
			pmm = -pmm * fact * somx2
			fact = fact + 2

	if L == M:
		return pmm
	else:
		pmmp1 = x * (2 * M + 1) * pmm
		if L == M + 1:
			return pmmp1
		else:
			pll = np.zeros_like(x)
			for i in range(M + 2, L + 1):
				pll = (x * (2 * i - 1) * pmmp1 - (i + M - 1) * pmm) / (i - M)
				pmm = pmmp1
				pmmp1 = pll
			return pll


def SphericalHarmonic(M, L, theta, phi):
	if M > 0:
		return math.sqrt(2.0) * KVal(M, L) * np.cos(M * phi) * AssociatedLegendre(M, L, np.cos(theta))
	elif M < 0:
		return math.sqrt(2.0) * KVal(-M, L) * np.sin(-M * phi) * AssociatedLegendre(-M, L, np.cos(theta))
	else:
		return KVal(0, L) * AssociatedLegendre(0, L, np.cos(theta))


def save_obj(mesh_path, verts):
	file = open(mesh_path, 'w')
	for v in verts:
		file.write('v %.4f %.4f %.4f\n' % (v[0], v[1], v[2]))
	file.close()


def sampleSphericalDirections(n):
	xv = np.random.rand(n, n)
	yv = np.random.rand(n, n)
	theta = np.arccos(1 - 2 * xv)
	phi = 2.0 * math.pi * yv

	phi = phi.reshape(-1)
	theta = theta.reshape(-1)

	vx = -np.sin(theta) * np.cos(phi)
	vy = -np.sin(theta) * np.sin(phi)
	vz = np.cos(theta)
	return np.stack([vx, vy, vz], 1), phi, theta


def getSHCoeffs(order, phi, theta):
	shs = []
	for n in range(0, order + 1):
		for m in range(-n, n + 1):
			s = SphericalHarmonic(m, n, theta, phi)
			shs.append(s)

	return np.stack(shs, 1)


def computePRT(mesh_path, n, order):
	mesh = trimesh.load(mesh_path, process=False)
	vectors_orig, phi, theta = sampleSphericalDirections(n)
	SH_orig = getSHCoeffs(order, phi, theta)

	w = 4.0 * math.pi / (n * n)

	origins = mesh.vertices
	normals = mesh.vertex_normals
	n_v = origins.shape[0]

	origins = np.repeat(origins[:, None], n, axis=1).reshape(-1, 3)
	normals = np.repeat(normals[:, None], n, axis=1).reshape(-1, 3)
	PRT_all = None
	for i in tqdm(range(n)):
		SH = np.repeat(SH_orig[None, (i * n):((i + 1) * n)], n_v, axis=0).reshape(-1, SH_orig.shape[1])
		vectors = np.repeat(vectors_orig[None, (i * n):((i + 1) * n)], n_v, axis=0).reshape(-1, 3)

		dots = (vectors * normals).sum(1)
		front = (dots > 0.0)

		delta = 1e-3 * min(mesh.bounding_box.extents)
		hits = mesh.ray.intersects_any(origins + delta * normals, vectors)
		nohits = np.logical_and(front, np.logical_not(hits))

		PRT = (nohits.astype(np.float) * dots)[:, None] * SH

		if PRT_all is not None:
			PRT_all += (PRT.reshape(-1, n, SH.shape[1]).sum(1))
		else:
			PRT_all = (PRT.reshape(-1, n, SH.shape[1]).sum(1))

	PRT = w * PRT_all

	# NOTE: trimesh sometimes break the original vertex order, but topology will not change.
	# when loading PRT in other program, use the triangle list from trimesh.
	return PRT, mesh.faces




def testPRT(in_path, out_path, watertight_bust_path, watertight_statue_path, n=40):

	if Path(os.path.join(out_path, 'errored.txt')).exists():
		errored = open(os.path.join(out_path, 'errored.txt')).readlines()
	else:
		errored = []

	watertight_bust = [x.strip('\n').split('/')[-1] for x in open(watertight_busts_path, 'r').readlines()][1:]
	watertight_statue = [x.strip('\n').split('/')[-1] for x in open(watertight_statues_path, 'r').readlines()][1:]

	folders = sorted([x for x in os.listdir(in_path) if not x.startswith('.') and not x.endswith('.txt')], key=int)
	for folder in folders:

		if not os.path.exists(os.path.join(out_path, folder)):
			os.mkdir(f'{out_path}/{folder}')

		reps = [x for x in os.listdir(os.path.join(in_path, folder)) if not x.startswith('.')]
		for rep in reps:

			if not os.path.exists(os.path.join(out_path, folder, rep)):
				os.mkdir(os.path.join(out_path, folder, rep))

			files = [x for x in os.listdir(os.path.join(in_path, folder, rep)) if not x.startswith('.') and not x.endswith('.png')]
			for file in files:
				if file in watertight_bust or file in watertight_statue:

					if ((Path(os.path.join(out_path, folder, rep, file[:-4] + '__bounce.txt')).exists() and Path(os.path.join(out_path, folder, rep, file[:-4] + '__face.npy')).exists()) or f'{folder}/{rep}/{file}' in errored):
						continue

					else:
						try:
							print(f'Processing file: {folder}/{rep}/{file}')
							PRT, F = computePRT(f'{in_path}/{folder}/{rep}/{file}', n, 2)
							np.savetxt(f'{out_path}/{folder}/{rep}/{file[:-4]}__bounce.txt', PRT, fmt='%.8f')
							np.save(f'{out_path}/{folder}/{rep}/{file[:-4]}__face.npy', F)
						except (AttributeError, MemoryError) as e:
							print(f'Adding {folder}/{rep}/{file} to errored due to {e}')
							errored.append(f'{folder}/{rep}/{file}')
							with open(f'{out_path}/errored.txt)', 'w') as f:
								for filepath in errored:
									f.write(f'{filepath}')

				elif file not in watertight_bust and file not in watertight_statue:
					if Path(os.path.join(out_path, folder, rep, file[:-4] + '__bounce.txt')).exists() and Path(os.path.join(out_path, folder, rep, file[:-4] + '__face.npy')).exists():
						print(f'Removing file: {folder}/{rep}/{file}')
						os.remove(os.path.join(out_path, folder, rep, file[:-4] + '__bounce.txt'))
						os.remove(os.path.join(out_path, folder, rep, file[:-4] + '__face.npy'))

					elif (f'{folder}/{rep}/{file}' in errored) and (file not in f'watertight_{rep.lower()}'):
						errored.remove(f'{folder}/{rep}/{file}')

					else:
						continue


#def testPRT(dir_path, n=40):
#	if dir_path[-1] == '/':
#		dir_path = dir_path[:-1]
#	sub_name = dir_path.split('/')[-1][:-4]
#	obj_path = os.path.join(dir_path, sub_name + '_100k.obj')
#	os.makedirs(os.path.join(dir_path, 'bounce'), exist_ok=True)

#	PRT, F = computePRT(obj_path, n, 2)
#	np.savetxt(os.path.join(dir_path, 'bounce', 'bounce0.txt'), PRT, fmt='%.8f')
#	np.save(os.path.join(dir_path, 'bounce', 'face.npy'), F)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	# Local
	#parser.add_argument('-i', '--input', type=str, default='/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset')
	#parser.add_argument('-o', '--out_dir', type=str, default='/Volumes/CKXZ 1/@City/363, FP/AISculpture/PIFuHD/DS-Related/prt_util')
	#parser.add_argument('-wb', '--wtight_bust', type=str, default='/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_BUSTS.txt')
	#parser.add_argument('-ws', '--wtight_statue', type=str, default='/Volumes/CKXZ 1/@City/363, FP/Dataset(s)/decimated_obj-dataset/watertight_STATUES.txt')
	# Server
	parser.add_argument('-i', '--input', type=str, default='/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_obj-dataset_vt')
	parser.add_argument('-o', '--out_dir', type=str, default='/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/prt_util')
	parser.add_argument('-wb', '--wtight_bust', type=str, default='/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_obj-dataset_vt/watertight_BUSTS.txt')
	parser.add_argument('-ws', '--wtight_statue', type=str, default='/home/enterprise.internal.city.ac.uk/adbb120/pifuhd/data/decimated_obj-dataset_vt/watertight_STATUES.txt')
	parser.add_argument('-n', '--n_sample', type=int, default=40,
						help='squared root of number of sampling. the higher, the more accurate, but slower')
	args = parser.parse_args()

	testPRT(args.input, args.out_dir, args.wtight_bust, args.wtight_statue, args.n_sample)