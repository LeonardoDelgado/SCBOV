# Archivo de configuracion de rotacion
from etxt_3 import plot_cloud,  alinear, get_plane_as_points, get_plane_from_points, skelet_by_plane, interest_points
import matplotlib.pyplot as plt
# from utilities_for_data1 import views_to_boxel
from open3d.open3d.io import read_point_cloud
# from sklearn.cluster import KMeans
import numpy as np
from skimage.measure import label, regionprops
# import math

###########x,y,z
x_filepath = '/home/leo/Datasets/Test_PCN_YCB/Test/Y/'
################  valor minimo 11761 ###############
################  valor maximo 13068 ###############
name = '11900'#'11900'#'12070'

pcd = read_point_cloud(x_filepath + int(name).__str__()+'.pcd')#.to_array()
x_np_pts = np.array(pcd.points)

x_np_ptsRRR = alinear(x_np_pts).astype('int')
skelet = skelet_by_plane(x_np_ptsRRR)
plot_cloud(skelet, grid_ = True, minimo = -20, maximo = 20)

inter_points = interest_points(x_np_ptsRRR, number_interest_points=15)
plot_cloud(inter_points, grid_ = True, minimo = -20, maximo = 20)

# plot_cloud(x_np_pts, grid_ = True)

# plot_cloud(x_np_ptsRR, grid_ = True, name = str(alpha*180/math.pi), minimo = -20, maximo = 20)

plot_cloud(x_np_ptsRRR, grid_ = True, name = 'z', minimo = -20, maximo = 20)

#Axis x
input()









# scale = 32
# views = getviews(x_np_ptsRRR,scale)
# voxel_cube = views_to_boxel(views,scale)
# skelet_ = skelet(voxel_cube,scale = scale)
# skelet_ = np.array(skelet_)
# plot_cloud(skelet_)

# kmeans = KMeans(n_clusters=10, random_state=0).fit(skelet_) # doctest: +SKIP
# a = kmeans.cluster_centers_
# plot_cloud(a)
