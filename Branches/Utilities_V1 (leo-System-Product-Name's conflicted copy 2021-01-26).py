#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 16 20:26:37 2020

@author: leo
"""
from skimage.measure import label, regionprops

#from open3d import read_point_cloud
import os

if os.name != 'nt':
    from open3d.open3d.io import read_point_cloud 
else:
    from open3d.open3d_pybind.io import read_point_cloud 
    

from mpl_toolkits.mplot3d import Axes3D 
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import copy
from numpy import sin, cos
from scipy import ndimage
import numpy as np
import math
from sklearn.neighbors import KDTree

class point_cloud_manipulation:
    def __init__(self, path):
        if type(path) ==type('ss'):
            pcd = read_point_cloud(path)
            self.point_cloud = np.array(pcd.points)
        else:
            self.point_cloud = path
        
        self.original = self.point_cloud
        self.operations = []
        self.last_moviment = []
        self.images = ''
    
    def alinear(self):
        self.center()
        Eje = self.get_axis()
        centroid2 = Eje[-1,:]
        betha  = math.atan(centroid2[1]/centroid2[2])
        self.rotationxyz(betha,0)
        self.center()
        Eje = self.get_axis()
        centroid2 = Eje[-1,:]
        alpha  = math.atan(centroid2[0]/centroid2[2])
        self.rotationxyz(-alpha,1)
        self.center()
        Eje = self.get_axis2()
        centroid2 = Eje[-1,:]
        theta  = math.atan(centroid2[0]/centroid2[1])
        self.rotationxyz(theta,2)
        self.center()
    
    def center(self):
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]
        cloud = [x_array,y_array,z_array]
        cloud2 = []
        for i,array in enumerate(cloud):
            centro = np.mean(array)
            cloud2.append(array-centro)
            self.last_moviment = (i,centro)
            self.operations.append((i,centro))
        self.point_cloud = np.array(cloud2).T
    
    def get_axis(self):
        pca = PCA(n_components = 1)
        pca.fit(self.point_cloud)
        eig_vec = pca.components_
        centroid = np.mean(self.point_cloud, axis=0)
        segments = np.arange(-10, 10)[:, np.newaxis] * eig_vec
        Eje = (centroid + segments)
        return Eje
    
    def get_axis2(self):
        pca = PCA(n_components = 2)
        pca.fit(self.point_cloud)
        eig_vec = pca.components_
        centroid = np.mean(self.point_cloud, axis=0)
        segments = np.arange(-10, 10)[:, np.newaxis] * eig_vec[1,:]
        Eje = (centroid + segments)
        return Eje        
    
    def rotationxyz(self,theta,axis):
        new_points = []
        self.last_moviment = (axis + 3, theta)
        self.operations.append((axis + 3, theta))
        for point in list(self.point_cloud):
            if axis == 0:
                new_points.append([point[0],
                                   self.coszero(theta)*point[1]-self.sinzero(theta)*point[2],
                                   self.sinzero(theta)*point[1]+self.coszero(theta)*point[2]])
            elif axis == 1:
                new_points.append([self.coszero(theta)*point[0]+self.sinzero(theta)*point[2],
                                   point[1],
                                   self.coszero(theta)*point[2]-self.sinzero(theta)*point[0]])
            elif axis == 2:
                new_points.append([self.coszero(theta)*point[0]-self.sinzero(theta)*point[1],
                                   self.coszero(theta)*point[1]+self.sinzero(theta)*point[0],
                                   point[2]])
        self.point_cloud = np.array(new_points)
    
    
    def coszero(self,x,lower_number = 0.001):
        y = cos(x)
        y =  ( 0.0 if abs(y) < lower_number else y )
        
        return y
        
    def sinzero(self,x,lower_number = 0.001):
        y = sin(x)
        y =  ( 0.0 if abs(y) < lower_number else y )
        return y
    
    def goback(self,steps):
        max_steps = len(self.operations)
        
        if steps == 0:
            steps = max_steps
            
        if steps <= max_steps:
            for step in range(steps):
                movement = self.operations.pop()
                type_of_movement = movement[0]
                amoung = movement[1]
                if type_of_movement < 3: #is traslation movement
                        x_array = self.point_cloud[:, 0]
                        y_array = self.point_cloud[:, 1]
                        z_array = self.point_cloud[:, 2]
                        if type_of_movement == 0:
                            x_array += amoung
                        elif type_of_movement == 1:
                            y_array += amoung
                        elif type_of_movement == 2:
                            z_array += amoung
                        self.point_cloud = np.array([x_array,y_array,z_array]).T
                elif type_of_movement >2  and type_of_movement < 6:
                    self.rotationxyz(-amoung, type_of_movement-3)
                    self.last_moviment = []
                    self.operations.pop()
                elif type_of_movement >5 and type_of_movement < 9:
                    self.scale(amoung,type_of_movement-6)
                    self.last_moviment = []
                    self.operations.pop()
        else:
            print('hay mas pasos que historial trate un numero menor o igual a: ', max_steps)
                
                
    def interest_points(self, number_interest_points):
        self.skelet_dumy = self.skelet_by_plane()
        inter_points = KMeans(n_clusters = number_interest_points).fit(self.skelet_dumy).cluster_centers_
        return skl_obj(inter_points)
    
    
    def skelet_by_plane(self):
        history = len(self.operations)
        self.alinear()
        skelet = []
        min_points = (self.point_cloud.shape[0]/100)
        point_cloud_asint = self.point_cloud.astype('int')
        for j in range(3):
            array_asint = point_cloud_asint[:,j]
            min_array = np.min(array_asint)
            max_array = np.max(array_asint)
            for i in range(min_array,max_array,1):
                points_plane = self.get_plane_as_points(i, axis=j)
                plane,minimo_1,minimo_2, skeleto = get_plane_from_points(points_plane,min_points,i,axis=j)
                if skeleto != []:
                    if skelet == []:                
                        skelet = skeleto
                    else:
                        skelet = np.concatenate((skelet,skeleto))
                        
        skelet = goback_s(skelet, len(self.operations)-history ,self.operations)
        self.goback(len(self.operations)-history)
        return skelet
            
        
    
    def get_plane_as_points(self,deep, axis=0):
        #point clouds in format X,Y,Z
        x_array = self.point_cloud[:,0].astype('int')
        y_array = self.point_cloud[:,1].astype('int')
        z_array = self.point_cloud[:,2].astype('int') 
        if axis == 0:
            mask = x_array == deep
        elif axis == 1:
            mask = y_array == deep
        else:
            mask = z_array == deep
        return self.point_cloud[mask,:].astype('int') 
    
    def mov_xyz(self, amoung, type_of_movement):
        self.last_moviment = (type_of_movement, -amoung)
        self.operations.append((type_of_movement, -amoung))
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]
        if type_of_movement == 0:
            x_array += amoung
        elif type_of_movement == 1:
            y_array += amoung
        elif type_of_movement == 2:
            z_array += amoung
        self.point_cloud = np.array([x_array,y_array,z_array]).T   
        
    def mov(self, amoung, type_of_movement):
        if type_of_movement < 3:
            self.mov_xyz(amoung[0],type_of_movement)
        elif type_of_movement >2 and type_of_movement < 6:
            self.rotationxyz(amoung[1],type_of_movement-3)
            
    def scale(self, amoung, type_of_movement):
        self.last_moviment = (6+type_of_movement, 1/amoung)
        self.operations.append((6+type_of_movement, 1/amoung))
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]
        if type_of_movement == 0:
            x_array *= amoung
        elif type_of_movement == 1:
            y_array *= amoung
        elif type_of_movement == 2:
            z_array *= amoung
        self.point_cloud = np.array([x_array,y_array,z_array]).T   
            
        
    def get_shell(self):
        scale = 35
        history = len(self.operations)
        self.alinear()
        self.getviews(scale)
        voxel = self.views_to_boxel(scale)
        self.point_cloud = voxel_to_cloud(voxel).astype('float')
        
        self.goback(len(self.operations)-history)
        
    
    def scalesize(self, scale = 255):
    #--------------------------------------set home
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]
        cloud = [x_array,y_array,z_array]
        for i,array in enumerate(cloud):
            minimo = np.min(array)
            if minimo<0:
                self.mov_xyz(-minimo,i)
            else:
                if minimo > 0:
                    self.mov_xyz(-minimo,i)

        #-------------------------------------- scale
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]
        x_max = np.max(x_array)
        y_max = np.max(y_array)
        z_max = np.max(z_array)
        
        temp = np.max([x_max,y_max,z_max])
        factor_scala = (scale-1)/temp           # El menos 1 es para que el rango este entre [0,(sclae -1)]
        self.scale(factor_scala,0)
        self.scale(factor_scala,1)
        self.scale(factor_scala,2)
        x_array = self.point_cloud[:, 0]
        y_array = self.point_cloud[:, 1]
        z_array = self.point_cloud[:, 2]

        #--------------------------------------- center
        for i,array in enumerate([x_array,y_array,z_array]):
            maximo = np.max(array)
            if maximo < scale-1:
                temp =(((scale-1)/2)-(maximo/2))
                self.mov_xyz(temp,i)
    #used to clear shell


    
    def getviews(self,scale=60):
        image = np.zeros((scale, scale,6))
        self.scalesize(scale)
        voxel_cube = pcdtovoxel(self.point_cloud,scale)
        image[:,:,0] = getimagefromvoxel(voxel_cube, scale, 0)
        image[:,:,1] = getimagefromvoxel(voxel_cube, scale, 1)
        image[:,:,2] = getimagefromvoxel(voxel_cube, scale, 2)
        image[:,:,3] = getimagefromvoxel(voxel_cube, scale, 3)
        image[:,:,4] = getimagefromvoxel(voxel_cube, scale, 4)
        image[:,:,5] = getimagefromvoxel(voxel_cube, scale, 5)
        self.images = image #, center, factor_scala
        
    def views_to_boxel(self,size):
        vistas = self.images.shape[2]
        voxel_cube = crear_cubo(size)
        for vista in range(vistas):
            if vista < 6:
                voxel_cube = add_view_to_cube(voxel_cube,self.images[:,:,vista],size,vista)       
        return voxel_cube
    
    def sample_pc(self,max_dis, sample = 0.1):
        
        high = self.point_cloud.shape[0]
        size = int(high*sample)
        sampleo = np.random.randint(0,high,size)
        point_cloud_sampled = self.point_cloud[sampleo]
        tree_envioroment = KDTree(self.point_cloud, leaf_size=10)
        self.sub_clouds = []
        for i in range(len(sampleo)):
            ind = tree_envioroment.query_radius(point_cloud_sampled[i,:].reshape(1,-1), r=max_dis)   
            self.sub_clouds.append((self.point_cloud[list(ind)],point_cloud_sampled[i,:]))
    def set_home(self,point):
        x = point[0]
        y = point[1]
        z = point[2]
        self.mov_xyz(-x, 0)
        self.mov_xyz(-y, 1)
        self.mov_xyz(-z, 2)
##################################################################################
class skl_obj:  
    def __init__(self, skeleto):

        self.skeleto = skeleto
        self.centro =  self.centro_()
        
        
    def centro_(self):
        mean = np.mean(self.skeleto, axis=0)
        tree_skeleto = KDTree(self.skeleto, leaf_size = 1)
        dist, ind = tree_skeleto.query(mean.reshape(1,-1), k=1)
        
        return self.skeleto[ind,:].reshape(1,3)


class its_obj:
    def __init__(self, world_cloud, query_cloud,  points, nits = 5):
        self.world_cloud = point_cloud_manipulation(world_cloud)
        self.world_cloud.get_shell()
        self.world_cloud = self.world_cloud.point_cloud
        self.query_cloud = query_cloud
        self.points = points
        self.nits = nits
        query_cloud_pcm = point_cloud_manipulation(self.query_cloud)
        self.query_skeleto_ = query_cloud_pcm.interest_points(self.points)
        self.query_skeleto = self.query_skeleto_.skeleto
        self.its_()    

        
    def its_(self):
        """
        Parameters
        ----------
        world_cloud : point cloud as numpy array [points,xyz]
        this point cloud represent the envioroment where the task is going to be serched
            
        query_cloud : point cloud as numpy arry
            This cloud represent the object which affordance is goin to be obteined
    
        points : int
            this variable
        Returns
        -------
        None.
        """
        leaf = int(.05*self.world_cloud.shape[0])
        if leaf < 50:
            leaf = 50
        tree_world_cloud = KDTree(self.world_cloud, leaf_size=leaf)
        self.its = []
        for i in range(self.query_skeleto.shape[0]):
            dist, ind = tree_world_cloud.query(self.query_skeleto[i,:].reshape(1,-1), k=leaf) 
            inter_points_its = KMeans(n_clusters = self.nits ).fit(self.world_cloud[ind[0,:]]).cluster_centers_
            self.its.append(inter_points_its)
        self.its_as_array =np.array(self.its).reshape(int(self.points*self.nits),3)
        #remap
        remap = []
        for i in range(self.its_as_array.shape[0]):
            dist, ind = tree_world_cloud.query(self.its_as_array[i,:].reshape(1,-1), k=1)
            remap.append(self.world_cloud[ind,:])
        self.its_as_array =np.array(remap).reshape(int(self.points*self.nits),3)
        self.its_as_array =np.unique(self.its_as_array,axis=0)
        self.centro_()
        self.normalized()
        
        
    def centro_(self):
        mean = np.mean(self.its_as_array, axis=0)
        its_tree = KDTree(self.its_as_array, leaf_size = 1)
        dist, ind = its_tree.query(mean.reshape(1,-1), k=1)
        
        self.centro = self.its_as_array[ind,:].reshape(1,3)
        
    
    def normalized(self):
        self.its_pcm = point_cloud_manipulation(self.its_as_array)
        # set ceter to home
        self.its_pcm.mov_xyz(-self.centro[0,0],0)
        self.its_pcm.mov_xyz(-self.centro[0,1],1)
        self.its_pcm.mov_xyz(-self.centro[0,2],2)
        self.its_tree =  KDTree(self.its_pcm.point_cloud, leaf_size = self.its_pcm.point_cloud.shape[0])
        self.its_norm_dist,ind = self.its_tree.query(np.array([0,0,0]).reshape(1,-1), k=self.its_pcm.point_cloud.shape[0])
        self.its_norm = self.its_pcm.point_cloud[ind]
        self.its_norm_dist = self.its_norm_dist[0,:]
        self.its_norm = self.its_norm[0,:,:]
        self.max_dis = np.max(self.its_norm_dist)
        self.punto_mas_lejano_norm = self.its_norm[-1,:].reshape((1,3))
        self.punto_mas_lejano = self.its_as_array[ind[0,-1],:].reshape((1,3))
        
def voxel_to_cloud(voxel_cube):
    array_x = []
    array_y = []
    array_z = []
    r,d,f = voxel_cube.shape
    for i in range(r):
        for j in range(d):
            for k in range(f):
                if voxel_cube[i,j,k] == 1:
                    array_x.append(i)
                    array_y.append(j)
                    array_z.append(k)
    return np.array([np.array(array_x), np.array(array_y), np.array(array_z)]).T    


def add_view_to_cube(voxel_cube,view_image,size,view):
    view_image = truncate_view(view_image,size)
    scale = size
    if view <= 2:
        for i in range(scale):
            for j in range(scale):
                if view == 0:
                    if view_image[i,j] >= 0:
                        voxel_cube[i,j,view_image[i,j]] = 1
                elif view == 1:
                    if view_image[i,j] >= 0:
                        voxel_cube[i,view_image[i,j],j] = 1
                elif view == 2:
                    if  view_image[i,j] >= 0:
                        voxel_cube[view_image[i,j],i,j] = 1
                else:
                    return voxel_cube
    else:
        for i in range(scale):
            for j in range(scale):
                if view == 3:
                    if view_image[i,j] >= 0:
                        voxel_cube[i,j,size-1-view_image[i,j]] = 1
                elif view == 4:
                    if view_image[i,j] >= 0:
                        voxel_cube[i,size-1-view_image[i,j],j]  = 1
                elif view == 5:
                    if  view_image[i,j] >= 0:
                        voxel_cube[size-1-view_image[i,j],i,j] = 1
                else:
                    return voxel_cube
    return voxel_cube    

def unique_rows(data):
    uniq = np.unique(data.view(data.dtype.descr * data.shape[1]))
    return uniq.view(data.dtype).reshape(-1, data.shape[1])

def truncate_view(view,size):
    offset = int(size*.1 + 1) 
    view = view - offset
    temp = np.logical_and(view>=size, view <= (size+4))
    view[temp] = size-1
    temp = np.logical_or(view<0,view>(size+4))
    view[temp] = -1
    view = view.astype(int)
    return view
def crear_cubo(size):
    cubo = np.zeros((size,size,size))
    return cubo

def pcdtovoxel(cloud,scale = 255, offset = False):
    voxel_cube = np.zeros((scale, scale, scale))
    cloud = [cloud[:,0],cloud[:,1],cloud[:,2]]
    x_array, y_array, z_array = cloud
    size_cloud = x_array.shape[0]
    for i in range(size_cloud):
        voxel_cube[int(x_array[i]), int(y_array[i]), int(z_array[i])] = 1
    return voxel_cube


def getimagefromvoxel(voxel_cube,scale = 255, view = 0, offset = True):
    if offset == True:
        offset = int(scale*.1 + 1) 
    image = np.zeros((scale, scale))
    if view <= 2:
        for i in range(scale):
            for j in range(scale):
                for k in range(scale):
                    if view == 0:
                        if voxel_cube[i,j,k] == 1:
                            image[i,j] = k + offset
                            break
                    elif view == 1:
                        if voxel_cube[i,k,j] == 1:
                            image[i,j] = k + offset
                            break
                    elif view == 2:
                        if voxel_cube[k,i,j] == 1:
                            image[i,j] = k + offset
                            break
                    else:
                        return image
    else:
        for i in range(scale):
            for j in range(scale):
                cn = 0
                for k in range(scale-1,-1,-1):
                    if view == 3:
                        if voxel_cube[i,j,k] == 1:
                            image[i,j] = cn + offset
                            break
                    elif view == 4:
                        if voxel_cube[i,k,j] == 1:
                            image[i,j] = cn + offset
                            break
                    elif view == 5:
                        if voxel_cube[k,i,j] == 1:
                            image[i,j] = cn + offset
                            break
                    else:
                        return image
                    cn += 1
    return image 






def get_plane_from_points(point_cloud,min_points,deep,axis=0):
    #point clouds in format X,Y,Z
    x_array = point_cloud[:,0]
    y_array = point_cloud[:,1]
    z_array = point_cloud[:,2]   
    new_skeleto = []
    if axis == 0:
        plane, minimo_1, minimo_2, skeleto = plane_from_arrays(y_array,z_array,min_points,axis=axis)
        for elemento in skeleto:
            new_skeleto.append([deep,elemento[1],elemento[0]])
    elif axis == 1:
        plane, minimo_1, minimo_2, skeleto = plane_from_arrays(x_array,z_array,min_points,axis=axis)
        for elemento in skeleto:
            new_skeleto.append([elemento[1],deep, elemento[0]])
    else:
        plane, minimo_1, minimo_2, skeleto = plane_from_arrays(x_array,y_array,min_points,axis=axis)
        for elemento in skeleto:
            new_skeleto.append([elemento[1], elemento[0],deep])
    return plane, minimo_1, minimo_2, np.array(new_skeleto)
    
    
    
def plane_from_arrays(array_1,array_2,min_points,axis=0):
    skeleto = []
    minimo_1 = np.min(array_1)
    minimo_2 = np.min(array_2)
    if minimo_1 < 0:
        array_1 = array_1-minimo_1
    if minimo_2 < 0:
        array_2 = array_2-minimo_2 
    maximo_1 = np.max(array_1.astype('int'))
    maximo_2 = np.max(array_2.astype('int'))
    plane = np.zeros((maximo_1+1,maximo_2+1))#el mas uno es por que el array empieza en 0
    for i in range(array_1.shape[0]):
        plane[array_1.astype('int')[i],array_2.astype('int')[i]] = 1
    plane = ndimage.binary_fill_holes(plane)
    plane = label(plane)
    for region in regionprops(plane):
    # take regions with large enough areas
        if region.area >= min_points:
            image = region.image
            centroid = ndimage.center_of_mass(image)
            array_1_value = round(centroid[1])
            array_2_value = round(centroid[0])
            if minimo_1 < 0:
                array_1_value += minimo_2
            if minimo_2 < 0:
                array_2_value += minimo_1
            skeleto.append((array_1_value,array_2_value)) 
    return plane, minimo_1, minimo_2, skeleto
            
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    
def goback_s(point_cloud, steps, operations):
    max_steps = len(operations)
    
    if steps == 0:
        steps = max_steps
        
    if steps <= max_steps:
        for step in range(steps):
            movement = operations.pop()
            type_of_movement = movement[0]
            amoung = movement[1]
            if type_of_movement < 3: #is traslation movement
                    x_array = point_cloud[:, 0]
                    y_array = point_cloud[:, 1]
                    z_array = point_cloud[:, 2]
                    if type_of_movement == 0:
                        x_array += amoung
                    elif type_of_movement == 1:
                        y_array += amoung
                    elif type_of_movement == 2:
                        z_array += amoung
                    point_cloud = np.array([x_array,y_array,z_array]).T
            elif type_of_movement >2  and type_of_movement < 6:
                point_cloud = rotationxyz(point_cloud, -amoung, type_of_movement-3)
            elif type_of_movement >5  and type_of_movement < 9:
                point_cloud = scale(point_cloud, 1/amoung, type_of_movement-6)
    else:
        print('hay mas pasos que historial trate un numero menor o igual a: ', max_steps)
    return point_cloud

def scale(point_cloud, amoung, type_of_movement):
    x_array = point_cloud[:, 0]
    y_array = point_cloud[:, 1]
    z_array = point_cloud[:, 2]
    if type_of_movement == 0:
        x_array *= amoung
    elif type_of_movement == 1:
        y_array *= amoung
    elif type_of_movement == 2:
        z_array *= amoung
    return np.array([x_array,y_array,z_array]).T   
    
def rotationxyz(points,theta,axis):
    new_points = []
    for point in list(points):
        if axis == 0:
            new_points.append([point[0],
                               coszero(theta)*point[1]-sinzero(theta)*point[2],
                               sinzero(theta)*point[1]+coszero(theta)*point[2]])
        elif axis == 1:
            new_points.append([coszero(theta)*point[0]+sinzero(theta)*point[2],
                               point[1],
                               coszero(theta)*point[2]-sinzero(theta)*point[0]])
        elif axis == 2:
            new_points.append([coszero(theta)*point[0]-sinzero(theta)*point[1],
                               coszero(theta)*point[1]+sinzero(theta)*point[0],
                               point[2]])
    return np.array(new_points)     

def coszero(x,lower_number = 0.001):
    y = cos(x)
    y =  ( 0.0 if abs(y) < lower_number else y )
    
    return y
    
def sinzero(x,lower_number = 0.001):
    y = sin(x)
    y =  ( 0.0 if abs(y) < lower_number else y )
    return y


"""
    old version plot cloud
def plot_cloud(point_clouds, 
               name = '',
               label_x = 'x',
               label_y ='y',
               label_z ='z',
               minimo = '',
               maximo = '',
               grid_ = False, 
               save = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if minimo == '':
        minimo = np.min(point_clouds)
    if maximo == '':
        maximo = np.max(point_clouds)
    
    if len(point_clouds.shape) < 3:    
        point_cloud = point_clouds
        x_array = point_cloud[:,0]
        y_array = point_cloud[:,1]
        z_array = point_cloud[:,2]        
        ax.scatter(x_array, y_array, z_array, c = 'b', marker = '.')
    else:
        cmap = get_cmap(point_clouds.shape[0]*100)
        for ind in range(point_clouds.shape[0]):
                point_cloud = point_clouds[ind,:,:]
                print(point_cloud.shape)
                x_array = point_cloud[:,0]
                y_array = point_cloud[:,1]
                z_array = point_cloud[:,2]        
                ax.scatter(x_array, y_array, z_array, marker = '.', c = cmap(ind*40))        
    ax.set_xlabel(label_x)
    ax.set_xlim(minimo, maximo)
    ax.set_ylabel(label_y)
    ax.set_ylim((minimo, maximo))
    ax.set_zlabel(label_z)
    ax.set_zlim(minimo, maximo)
    
    if grid_ == False:
        ax.grid(False)
    # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.title(name)
    fig.show()
    if save == True:
        fig.savefig(name+'p.pdf')
"""


def plot_cloud(point_clouds, 
               name = '',
               label_x = 'x',
               label_y ='y',
               label_z ='z',
               minimo = '',
               maximo = '',
               grid_ = False, 
               save = False,
               cmap = [],
               alpha = []):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if len(point_clouds) < 2:    
        if minimo == '':
            minimo = np.min(point_clouds)
        if maximo == '':
            maximo = np.max(point_clouds)
        point_cloud = point_clouds[0]
        x_array = point_cloud[:,0]
        y_array = point_cloud[:,1]
        z_array = point_cloud[:,2]  
        if cmap == []:
            ax.scatter(x_array, y_array, z_array, c = 'b', marker = '.')
        else:
            ax.scatter(x_array, y_array, z_array, c = cmap[0], marker = '.')
        
    else:
        temp = point_clouds.pop()
        
        
        temp2 = copy.deepcopy(temp)
        for item in point_clouds:
            temp = np.concatenate((temp,item))

            
        if minimo == '':
            minimo = np.min(temp)
        if maximo == '':
            maximo = np.max(temp)
        point_clouds.append(temp2)    
            
        # cmap = get_cmap((len(point_clouds)+1)*80)
        
        for ind in range(len(point_clouds)):
                point_cloud = point_clouds[ind]
                x_array = point_cloud[:,0]
                y_array = point_cloud[:,1]
                z_array = point_cloud[:,2]
                if cmap == [] and alpha == []:
                    ax.scatter(x_array, y_array, z_array, marker = '.',c='r', alpha = .9) 
                elif cmap == []:
                    ax.scatter(x_array, y_array, z_array, marker = '.',c='r', alpha = alpha[ind]) 
                elif alpha == []:
                    ax.scatter(x_array, y_array, z_array, marker = '.',c = cmap[ind], alpha = .9)
                else:
                    ax.scatter(x_array, y_array, z_array, marker = '.',c = cmap[ind], alpha = alpha[ind])
    ax.set_xlabel(label_x)
    ax.set_xlim(minimo, maximo)
    ax.set_ylabel(label_y)
    ax.set_ylim((minimo, maximo))
    ax.set_zlabel(label_z)
    ax.set_zlim(minimo, maximo)
    
    if grid_ == False:
        ax.grid(False)
    # Hide axes ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    plt.title(name)
    fig.show()
    if save == True:
        fig.savefig(name+'p.pdf')

if __name__ == '__main__':
    from Utilities_V1 import point_cloud_manipulation as pcm
    from etxt_3 import plot_cloud
    x_filepath = '/home/leo/Datasets/Test_PCN_YCB/Test/Y/'
################  valor minimo 11761 ###############
################  valor maximo 13068 ###############
    name = '11900'#'11900'#'12070'
    # objeto_a = pcm(x_filepath + name + '.pcd')
    # eskelet = objeto_a.interest_points(15).skeleto
    # plot_cloud(eskelet, grid_ = True)
    # input()
    
    