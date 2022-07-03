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
    temp = view>size
    view[temp] = size
    view = view.astype(int)
    return view

def add_view_to_cube(voxel_cube,view_image,size,view):
    view_image = truncate_view(view_image,size)
    scale = size
    if view <= 2:
        for i in range(scale):
            for j in range(scale):
                if view == 0:
                    if view_image[i,j] > 0:
                        voxel_cube[i,j,view_image[i,j]] = 1
                elif view == 1:
                    if view_image[i,j] > 0:
                        voxel_cube[i,view_image[i,j],j] = 1
                elif view == 2:
                    if  view_image[i,j] > 0:
                        voxel_cube[view_image[i,j],i,j] = 1
                else:
                    return voxel_cube
    else:
        for i in range(scale):
            for j in range(scale):
                if view == 3:
                    if view_image[i,j] > 0:
                        voxel_cube[i,j,-view_image[i,j]] = 1
                elif view == 4:
                    if view_image[i,j] > 0:
                        voxel_cube[i,-view_image[i,j],j]  = 1
                elif view == 5:
                    if  view_image[i,j] > 0:
                        voxel_cube[-view_image[i,j],i,j] = 1
                else:
                    return voxel_cube

    return voxel_cube 

def views_to_boxel(views,size):
    vistas = views.shape[2]
    voxel_cube = crear_cubo(size)
    for vista in range(vistas):
        voxel_cube = add_view_to_cube(voxel_cube,views[:,:,vista],size,vista)
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

if __name__ == '__main__':
	datasets_directory = '/media/leo/Data/' 
	X_val = np.load(datasets_directory + 'X_val.npy')
	view_1, view_2, view_3, view_4, view_5, view_6 = split_views(X_val)
	print(view_1.shape)
	print(view_2.shape)
	print(view_3.shape)
	print(view_4.shape)
	print(view_5.shape)
	print(view_6.shape)






