#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 03:36:36 2020

@author: leo
"""
import numpy as np
from Utilities_V1 import plot_cloud
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KDTree
import open3d.open3d as o3d 
from Utilities_V1 import its_obj
from Utilities_V1 import point_cloud_manipulation as pcm
import copy
# import pyvista as pv

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    
def preprocess_point_cloud(pcd, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * .6
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, distance_threshold,
        o3d.registration.TransformationEstimationPointToPoint(False),
        4, [
            o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.registration.RANSACConvergenceCriteria(4000000, 500))
    return result



NAME = '8_9_2020Sample.npy'
scene = np.load(NAME)
tree_scene = KDTree(scene, leaf_size=10)              # doctest: +SKIP
distance = []


for i in range(scene.shape[0]):
    dist, ind = tree_scene.query(scene[i,:].reshape(1,-1), k=10) 
    distance.append(np.mean(dist[0,1:]))
eps = np.mean(distance)
min_samples = 4



clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(scene)
labels = clustering.labels_


query_cloud = scene[labels == 1]
world_cloud = scene[labels == 0]
points = 20
nits = 5

affordance = its_obj(query_cloud, world_cloud, points, nits)

pcd_its_norm = o3d.geometry.PointCloud()
t=pcm(affordance.its_norm)
pcd_its_norm.points = o3d.utility.Vector3dVector(t.point_cloud)


########enviroment cloud
x_filepath = '/home/leo/Datasets/Test_PCN_YCB/Test/Y/'
name1 = '12070'#'11985'#'11900'#'12070'
name2 = '12070'

obj1 = pcm(x_filepath+name1+'.pcd')
obj1.get_shell()


#################Sample point cloud
obj1.sample_pc(affordance.max_dis, sample = 0.1)
voxel_size = 2
r_result = []
for cloud, center in obj1.sub_clouds:
    plot_cloud([obj1.point_cloud,center.reshape((1,-1)),cloud],
               cmap  = ['y','k','b'],
               alpha = [ .05,1, .2 ])
    eviroment_query = pcm(cloud)
    eviroment_query.set_home(center)
    
    pcd_enviroment = o3d.geometry.PointCloud()
    pcd_enviroment.points = o3d.utility.Vector3dVector(eviroment_query.point_cloud)



    source_down, source_fpfh = preprocess_point_cloud(pcd_its_norm, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(pcd_enviroment, voxel_size)

    result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)
    r_result.append(result_ransac)
    print(result_ransac)
    
fitness_r = []
for result in r_result:
    fitness_r.append(result.fitness)
sortend_fitnes_arg = np.argsort(-np.array(fitness_r))

cloud, center = obj1.sub_clouds[sortend_fitnes_arg[0]]
eviroment_query = pcm(cloud)
eviroment_query.set_home(center)

pcd_enviroment = o3d.geometry.PointCloud()
pcd_enviroment.points = o3d.utility.Vector3dVector(eviroment_query.point_cloud)

draw_registration_result(pcd_its_norm, pcd_enviroment, r_result[sortend_fitnes_arg[0]].transformation)




    # threshold = 0.00002
    # trans_init = np.asarray([[1., 0., 0., 0.,],
    #                          [0., 1., 0., 0.,],
    #                          [0., 0., 1., 0.,],
    #                          [0., 0., 0., 1.,]])
    
    # # draw_registration_result(pcd_its_norm, pcd_enviroment, trans_init)
    # # evaluation = o3d.registration.evaluate_registration(pcd_its_norm, pcd_enviroment,
    #                                                         # threshold, trans_init)

    
    # print("Apply point-to-point ICP")
    # reg_p2p = o3d.registration.registration_icp(
    #      pcd_enviroment, pcd_its_norm, threshold,trans_init,
    #      o3d.registration.TransformationEstimationPointToPoint(),
    #      o3d.registration.ICPConvergenceCriteria(max_iteration = 2000))
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    # print("")
    # draw_registration_result(pcd_its_norm, pcd_enviroment, reg_p2p.transformation)

# plot_cloud([obj1.point_cloud])
# plot_cloud([obj1.point_cloud,sample_points],
#            cmap  = ['b','r'],
#            alpha = [0.3, 1 ])
# plot_cloud([affordance.world_cloud,affordance.query_cloud],
#            cmap  = ['y','b'],
#            alpha = [0.3, 1 ])

# plot_cloud([affordance.query_skeleto_.skeleto])
# plot_cloud([affordance.its_norm])
# plot_cloud([#affordance.world_cloud,
#             affordance.query_skeleto_.skeleto,
#             affordance.query_skeleto_.centro,
#             affordance.its_as_array,
#             affordance.centro,
#             affordance.world_cloud,
#             affordance.query_cloud,
#             affordance.punto_mas_lejano],
#            cmap  = ['g','k','r','k','y','b','k'],
#            alpha = [0.5, 1 ,0.7,1,.1,.02,1])

# plot_cloud([#affordance.world_cloud,
#             affordance.query_skeleto_.skeleto,
#             affordance.world_cloud,
#             affordance.query_cloud],
#            cmap  = ['g','y','b'],
#            alpha = [0.5,1,.02])


########################################################################################################
# x_filepath = '/home/leo/Datasets/Test_PCN_YCB/Test/Y/'
# name1 = '12070'#'11985'#'11900'#'12070'
# name2 = '12070'

# obj1 = pcm(x_filepath+name1+'.pcd')
# obj1.alinear()
# temp = obj1.point_cloud
# sklt = obj1.interest_points(20)
# sklt_dumy  = obj1.skelet_dumy


# plot_cloud([temp,sklt.skeleto,sklt_dumy],
#            cmap  = ['y','k','b'],
#            alpha = [ .05,1, .2 ])

# plot_cloud([sklt.skeleto],
#            cmap  = ['k'],
#            alpha = [ 1])

# plot_cloud([sklt_dumy],
#            cmap  = ['b'],
#            alpha = [ 0.5 ])
##########################################################################################################
input()
