import numpy as np

def multi_view_to_single_view(data,views = 6):
    new_data = []
    for number in range(data.shape[0]):
        for view in range(views):
            new_data.append(data[number,:,:,view])
    new_data = np.array(new_data)
    elementos,size1,size2 = new_data.shape
    return new_data.reshape((elementos,size1,size2,1))

def split_views(data,views = 6):
    new_data = []
    for view in range(views):
        new_data.append([])
    for number in range(data.shape[0]):
        for view in range(views):
            new_data[view].append(data[number,:,:,view])
    for view in range(views):
        array = np.array(new_data[view])    
        elementos,size1,size2 = array.shape
        new_data[view] = array
    return new_data[0].reshape((elementos,size1,size2,1)), new_data[1].reshape((elementos,size1,size2,1)), new_data[2].reshape((elementos,size1,size2,1)), new_data[3].reshape((elementos,size1,size2,1)), new_data[4].reshape((elementos,size1,size2,1)), new_data[5].reshape((elementos,size1,size2,1))

def crear_cubo(size):
    cubo = np.zeros((size,size,size))
    return cubo

def truncate_view(view,size):
    offset = int(size*.1 + 1) 
    view = view - offset
    temp = np.logical_and(view>=size, view <= (size+4))
    view[temp] = size-1
    temp = np.logical_or(view<0,view>(size+4))
    view[temp] = -1
    view = view.astype(int)
    return view


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

def views_to_boxel(views,size):
    vistas = views.shape[2]
    voxel_cube = crear_cubo(size)
    for vista in range(vistas):
        if vista < 6:
            voxel_cube = add_view_to_cube(voxel_cube,views[:,:,vista],size,vista)

    for vista in range(int(vistas/2)):
        if vista < 6:
            voxel_cube = filter(view_image = views[:,:,vista],
                view = vista,
                voxel_cube = voxel_cube,
                view_image_2 = views[:,:,vista+2])
    
    return voxel_cube

def boxel_to_cloud(voxel_cube):
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
    return np.array(array_x), np.array(array_y), np.array(array_z)

def filter(view_image,view_image_2, view,voxel_cube):
    size = view_image.shape[0]
    view_image = truncate_view(view_image,size)
    view_image_2 = truncate_view(view_image_2,size)
    for i in range(size):
        for j in range(size):
            if view_image[i,j] == -1 and view_image_2[i,j] == -1:
                if view == 0:
                    pass
                    voxel_cube[i,j,:] = 0
                elif view == 1:
                    pass
                    voxel_cube[i,:,j] = 0
                elif view == 2:
                    voxel_cube[:,i,j] = 0
                else:
                    return voxel_cube    
    return voxel_cube 

def dumi_voxel_cube(views,size):
    vistas = views.shape[2]
    voxel_cube = np.ones((size,size,size))
    for vista in range(int(vistas/2)):
        if vista < 6:
            voxel_cube = filter(view_image = views[:,:,vista],
                view = vista,
                voxel_cube = voxel_cube,
                view_image_2 = views[:,:,vista+3])
    return voxel_cube
    
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    datasets_directory = '/home/leo/Datasets/ShapeNet 32/Validation/Y/' 
    X_val = np.load(datasets_directory + '03001627_4ecb13c40d55bc24fbcb0345af819bcb_0.npy')
    voxel_cube = views_to_boxel(X_val,32)
    print(voxel_cube.shape)
    x_array, y_array, z_array = boxel_to_cloud(voxel_cube)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x_array, y_array, z_array, c = 'r', marker = '.')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_zaxis().set_visible(False)
    ax.set_xticks(range(33))
    ax.set_yticks(range(33))
    ax.set_zticks(range(33))
    plt.axis('off')    
    plt.show()

    







