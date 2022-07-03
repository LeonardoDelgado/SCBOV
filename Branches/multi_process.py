import multiprocessing
import numpy as np
import os  
if os.name != 'nt':
    from open3d.open3d.io import read_point_cloud 
    import open3d.open3d as o3d 
else:
    from open3d.open3d_pybind.io import read_point_cloud 
    import open3d.open3d_pybind as o3d 
    
from Utilities_V1 import point_cloud_manipulation as pcm


def ran_xyz(point_cloud, axis, q):
    v_ran_max = np.max(point_cloud[:,axis])
    v_ran_min = np.min(point_cloud[:,axis])
    v_ran = (v_ran_min,v_ran_max)
    q.put(v_ran)

if __name__ == "__main__":
    if os.name != 'nt':
        x_filepath = '/home/leo/Datasets/Test_PCN_YCB/Test/Y/'
    else:
        x_filepath = 'D:/Y/'
        # x_filepath = 'D:/interaction-tensor-master/Testing/data/'
    name1 = '11985'#'11985'#'11900'#'12070'
    # name1 = 'kitchen5_d'
    name2 = '12070'
    obj = pcm(x_filepath+name1+'.pcd')
    point_cloud = obj.point_cloud
    x_array = point_cloud[:,0]
    y_array = point_cloud[:,0]
    z_array = point_cloud[:,0]
    
    for ind_x in x_array:
        for ind_y in y_array:
            with Pool(processes=4) as pool:
    
    
    
    q = multiprocessing.Queue()
    p1 = multiprocessing.Process(target = ran_xyz,args=(point_cloud,0, q))
    p2 = multiprocessing.Process(target = ran_xyz,args=(point_cloud,1, q))
    p3 = multiprocessing.Process(target = ran_xyz,args=(point_cloud,2, q))
    p1.start()
    p2.start()
    p3.start()
    p1.join()
    p2.join()
    p3.join()
    while q.empty() != True:
        print(q.get())
