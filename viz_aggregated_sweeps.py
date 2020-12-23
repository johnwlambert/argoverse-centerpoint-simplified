
import glob
import pdb

from colour import Color

import mayavi
import numpy as np
from mayavi import mlab

from argoverse.visualization.mayavi_utils import draw_lidar, plot_points_3D_mayavi, draw_coordinate_frame_at_origin
from argoverse.utils.pkl_utils import load_pkl_dictionary
#from argoverse.utils import mayavi_wrapper

"""
Argoverse:
egovehicle frame

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/a073e840-6319-3f0b-843e-f6dccdcc7b77_lidar_PC_315974277719628000.ply.pkl
argoverse
Median:  [ 1.1709181 -3.1072214  1.392003  16.       ]
Mean:  [ 3.9762316  8.334666   2.0852027 25.063263 ]
Shape:  (451879, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/c9d6ebeb-be15-3df8-b6f1-5575bea8e6b9_lidar_PC_315973076019747000.ply.pkl
argoverse
Median:  [ 0.9060904 -2.527429   1.370414   4.       ]
Mean:  [1.5276272 4.161011  1.5626434 7.039471 ]
Shape:  (450988, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/7d37fc6b-1028-3f6f-b980-adb5fa73021e_lidar_PC_315968404723417000.ply.pkl
argoverse
Median:  [1.4137762  0.13636465 1.5030621  2.        ]
Mean:  [ 3.1302218  -0.55806845  1.8599783   7.1770077 ]
Shape:  (391463, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/10b8dee6-778f-33e4-a946-d842d2d9c3d7_lidar_PC_315968230420341000.ply.pkl
argoverse
Median:  [1.1443    0.5824372 1.3939385 9.       ]
Mean:  [ 3.0120149  1.6471171  1.8866845 16.949873 ]
Shape:  (443036, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/15c802a9-0f0e-3c87-b516-a3fa02f1ecb0_lidar_PC_315970768520053000.ply.pkl
argoverse
Median:  [2.390792  1.5347978 1.2803922 7.       ]
Mean:  [-0.4840033  4.400695   2.0194175 13.700847 ]
Shape:  (417903, 4)
"""


"""
nuscenes

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/fcb1422b6f6241939953d032f2cc2ce8.pkl
nuscenes
Median:  [-7.5751872e-05 -1.9433178e+00 -1.5570810e+00  9.0000000e+00]
Mean:  [-0.09244746 -1.6678356  -0.5153281  21.870104  ]
Shape:  (280076, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/00bf6d20450748048caf2959e330069b.pkl
nuscenes
Median:  [ 0.21140242 -0.18297504 -0.9678898   9.        ]
Mean:  [-0.6390955   1.6927952  -0.36515146 16.511572  ]
Shape:  (244587, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/ad8905836f364a87a7233eb0aef915ea.pkl
nuscenes
Median:  [-0.47689497 -0.27486348 -1.6717255   8.        ]
Mean:  [-1.0947456 -0.7767361 -0.836858  14.227299 ]
Shape:  (256557, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/36335281ce184e9e92d39662f8de2d20.pkl
nuscenes
Median:  [-5.9077668e-01  4.9452839e-05 -1.2800446e+00  1.1000000e+01]
Mean:  [-2.172643  -0.6909648 -0.6920489 17.38631  ]
Shape:  (270546, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/bd2e11f548474f67be4d4c033714d68b.pkl
nuscenes
Median:  [ 0.36082348 -1.2619897  -1.4404428  21.        ]
Mean:  [ 1.2701099  -1.6474319  -0.44492245 29.693943  ]
Shape:  (276423, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/9fd8ae00df104af290529a90cc5068b4.pkl
nuscenes
Median:  [ 1.2253922 -2.234418  -1.620625   3.       ]
Mean:  [-0.06452195 -3.5163016  -0.6955196  11.952485  ]
Shape:  (243567, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_pkl/daf998de0f354de4850f63d06a79d2b5.pkl
nuscenes
Median:  [ 2.1924432e-03 -1.9839654e+00 -1.6464293e+00  1.4000000e+01]
Mean:  [-0.5723663  -0.86392725 -0.9772423  18.419342  ]
Shape:  (262676, 4)
"""


"""
argoverse lidar frame
On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/22160544_2216_2216_2216_722161741824_lidar_PC_315966725320351000.ply.pkl
argoverse
Median:  [-0.52125907  1.1801276  -0.05036405  5.        ]
Mean:  [-0.5189586  1.3347757  0.351572  11.643615 ]
Shape:  (453899, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/d60558d2-d1aa-34ee-a902-e061e346e02a_lidar_PC_315971348820415000.ply.pkl
argoverse
Median:  [-1.2976315   0.5365444  -0.20044369 13.        ]
Mean:  [ 4.761806   -2.7998054   0.64692605 20.272867  ]
Shape:  (455590, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/cb0cba51-dfaf-34e9-a0c2-d931404c3dd8_lidar_PC_315972708119382000.ply.pkl
argoverse
Median:  [ 0.02152862 -1.7140266  -0.11145579  4.        ]
Mean:  [0.8562865 2.622221  0.2112681 8.421499 ]
Shape:  (460831, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/22160544_2216_2216_2216_722161741824_lidar_PC_315966732920272000.ply.pkl
argoverse
Median:  [ 0.0497084   1.7379792  -0.09052764  6.        ]
Mean:  [ 1.0522554   0.14329916  0.32596755 12.246333  ]
Shape:  (453602, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/22160544_2216_2216_2216_722161741824_lidar_PC_315966717719431000.ply.pkl
argoverse
Median:  [ 0.02824224  0.35116634 -0.14592077  8.        ]
Mean:  [-0.22823165 -2.3420477   0.5131193  18.548702  ]
Shape:  (463947, 4)

On /Users/jlambert/Downloads/argoverse-centerpoint-simplified/argoverse_pkl/45753856_4575_4575_4575_345754906624_lidar_PC_315969092619890000.ply.pkl
argoverse
Median:  [ 0.22553408 -1.4993582  -0.14902312 21.        ]
Mean:  [ 1.1359622   2.16658     0.03450739 32.258076  ]
Shape:  (471597, 4)

"""



def main():
	""" """
	#dname = 'argoverse'
	dname = 'nuscenes'
	if dname == 'nuscenes':
		nsweeps = 10
	elif dname == 'argoverse':
		nsweeps = 5
	wildcard = f'/Users/jlambert/Downloads/argoverse-centerpoint-simplified/{dname}_pkl/*.pkl'
	#pdb.set_trace()
	fpaths = glob.glob(wildcard)
	fpaths.sort()
	for fpath in fpaths:
		pkl_data = load_pkl_dictionary(fpath)
		print(f'On {fpath}')
		bgcolor = (0,0,0)
		fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))

		colormap = 'spectral'

		# colors = np.array(
		# 	[
		# 		[255,255,255], # white
		# 		[255,0,0], # red
		# 		[255,255, 0], # yellow
		# 		[0,255,0], # green
		# 		[0,0,255], # blue
		# 	]) / 255

		colors = np.array(
			[[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), nsweeps)]
		).squeeze()

		for i, point_cloud in enumerate(pkl_data['sweep_points_list']):
			#draw_lidar(point_cloud, colormap, fig)
			color = tuple(colors[i])
			fig = plot_points_3D_mayavi(
				points=point_cloud,
				fig=fig,
				fixed_color=color
			)
			fig = draw_coordinate_frame_at_origin(fig)

		all_points = np.concatenate(pkl_data['sweep_points_list'],axis=0)
		print(dname)
		print('Median: ', np.median(all_points, axis=0) )
		print('Mean: ', np.mean(all_points, axis=0) )
		print('Shape: ', all_points.shape)
		print()

		circle_point_cloud = get_3d_circle_points( np.zeros(2), radius=65, n_pts=int(1e6), height=-1.5)
		fig = plot_points_3D_mayavi(
			points=circle_point_cloud,
			fig=fig,
			fixed_color=(1, 1, 1)
		)

		mlab.view(azimuth=180)
		mlab.show()


def get_circle_points(center: np.ndarray, radius: float, n_pts: int = 100) -> np.ndarray:
	""" """
	assert center.size == 2
	assert isinstance(radius, float) or isinstance(radius, int)

	theta = np.linspace(0, 2*np.pi, n_pts)
	x = center[0] + radius * np.cos(theta)
	y = center[1] + radius * np.sin(theta)
	return np.hstack([ x.reshape(-1,1), y.reshape(-1,1) ])


def get_3d_circle_points(center, radius, n_pts, height):
	""" """
	circle_pts_xy = get_circle_points(center, radius, n_pts)
	# now, convert to 3d
	circle_pts_xyz = np.ones((n_pts,3)) * height
	circle_pts_xyz[:,:2] = circle_pts_xy
	return circle_pts_xyz


def vis_voxels():
	""" """
	pkl_fpath = "/Users/jlambert/Downloads/argoverse-centerpoint-simplified/nuscenes_prediction.pkl"
	pkl_data = load_pkl_dictionary(pkl_fpath)

	for token, sweep_output in pkl_data.items():

		coordinates = sweep_output['input_coordinates']
		voxels = sweep_output['input_voxels']
		pdb.set_trace()



if __name__ == '__main__':
	#vis_aggregated_sweeps()
	vis_voxels()

