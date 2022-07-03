import numpy as np
import pcl
from matplotlib import pyplot as plt
from matplotlib.image import imsave
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
 

def scalesize(x_np_pts, scale = 255):
    #--------------------------------------set home
    x_array = x_np_pts[:, 0]
    y_array = x_np_pts[:, 1]
    z_array = x_np_pts[:, 2]
    cloud = [x_array,y_array,z_array]
    cloud2 = []
    for array in cloud:
        minimo = np.min(array)
        if minimo<0:
            cloud2.append(array+abs(minimo)) 
        else:
            if minimo > 0:
                cloud2.append(array-minimo)
            else:
                cloud2.append(array) 
    x_array, y_array, z_array = cloud2
    #-------------------------------------- scale
    x_max = np.max(x_array)
    y_max = np.max(y_array)
    z_max = np.max(z_array)
    temp = np.max([x_max,y_max,z_max])
    factor_scala = scale/temp
    x_array = np.round(x_array*factor_scala)
    y_array = np.round(y_array*factor_scala)
    z_array = np.round(z_array*factor_scala)
    #--------------------------------------- center
    cloud = [x_array,y_array,z_array]
    cloud2 = []
    for array in cloud:
        maximo = np.max(array)
        if maximo < scale:
            cloud2.append(array + ((scale/2)-(maximo/2)))
        else:
            cloud2.append(array)

    return cloud2

def pcdtovoxel(cloud,scale = 255):
    voxel_cube = np.zeros((scale + 1,scale + 1,scale + 1)) # El mas uno es por que va de 0 a scale - 1
    x_array, y_array, z_array = cloud
    size_cloud = x_array.shape[0]
    for i in range(size_cloud):
        voxel_cube[int(x_array[i]), int(y_array[i]), int(z_array[i])] = 1
    return voxel_cube


    

def getimagefromvoxel(voxel_cube,scale = 255, view = 0):
    image = np.zeros((scale + 1,scale + 1))
    if view <= 2:
        for i in range(scale+1):
            for j in range(scale+1):
                for k in range(scale+1):
                    if view == 0:
                        if voxel_cube[i,j,k] == 1:
                            image[i,j] = k
                            break
                    elif view == 1:
                        if voxel_cube[i,k,j] == 1:
                            image[i,j] = k
                            break
                    elif view == 2:
                        if voxel_cube[k,i,j] == 1:
                            image[i,j] = k
                            break
                    else:
                        return image
    else:
        for i in range(scale+1):
            for j in range(scale+1):
                cn = 0
                for k in range(scale,-1,-1):
                    cn += 1
                    if view == 3:
                        if voxel_cube[i,j,k] == 1:
                            image[i,j] = cn
                            break
                    elif view == 4:
                        if voxel_cube[i,k,j] == 1:
                            image[i,j] = cn
                            break
                    elif view == 5:
                        if voxel_cube[k,i,j] == 1:
                            image[i,j] = cn
                            break
                    else:
                        return image
    return image 

def getviews(x_np_pts,scale):
    scale = scale - 1
    image = np.zeros((scale + 1,scale + 1,6))
    cloud = scalesize(x_np_pts, scale)
    voxel_cube = pcdtovoxel(cloud,scale)
    image[:,:,0] = getimagefromvoxel(voxel_cube, scale, 0)
    image[:,:,1] = getimagefromvoxel(voxel_cube, scale, 1)
    image[:,:,2] = getimagefromvoxel(voxel_cube, scale, 2)
    image[:,:,3] = getimagefromvoxel(voxel_cube, scale, 3)
    image[:,:,4] = getimagefromvoxel(voxel_cube, scale, 4)
    image[:,:,5] = getimagefromvoxel(voxel_cube, scale, 5)

    return image

def getviews_from_voxel_cube(voxel_cube,scale):
    scale = scale - 1
    image[:,:,0] = getimagefromvoxel(voxel_cube, scale, 0)
    image[:,:,1] = getimagefromvoxel(voxel_cube, scale, 1)
    image[:,:,2] = getimagefromvoxel(voxel_cube, scale, 2)
    image[:,:,3] = getimagefromvoxel(voxel_cube, scale, 3)
    image[:,:,4] = getimagefromvoxel(voxel_cube, scale, 4)
    image[:,:,5] = getimagefromvoxel(voxel_cube, scale, 5)

    return image


if __name__ == '__main__':
    #data = np.load('/media/leo/OLD/Descargas/deep_marching_cubes_data/voxel_shapenet_32x32x32_train.npy')
    #voxel_cube = abs(data[3000,:,:,:]-6)>0
    name = 'donut_poisson_009'
    x_filepath = '/media/leo/OLD/Descargas/grasp_database/grasp_database/'+name+'/pointclouds/_0_4_5_x.pcd'
    x_np_pts = pcl.load(x_filepath).to_array()
    scale = 32
    cloud = scalesize(x_np_pts, scale)
    voxel_cube = pcdtovoxel(cloud,scale)
    
    imagen1 = getimagefromvoxel(voxel_cube, scale, 0)
    imagen2 = getimagefromvoxel(voxel_cube, scale, 1)
    imagen3 = getimagefromvoxel(voxel_cube, scale, 2)
    imagen4 = getimagefromvoxel(voxel_cube, scale, 3)
    imagen5 = getimagefromvoxel(voxel_cube, scale, 4)
    imagen6 = getimagefromvoxel(voxel_cube, scale, 5)
    for i,imagen in enumerate([imagen1,imagen2,imagen3,imagen4,imagen5,imagen6]):
        imsave('vista'+ i.__str__() +'.png',imagen)

    
    
    x_array, y_array, z_array = cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
    plt.show()
    
    
