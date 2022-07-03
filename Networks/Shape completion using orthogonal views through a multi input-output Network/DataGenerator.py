import numpy as np
import keras

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,path,path_list_IDs, batch_size=32, dim=(32,32), n_channels=1, shuffle=True):
        'Initialization'
        f = open(path_list_IDs)
        list_IDs = f.read().splitlines()
        f.close()
        self.path = path
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def on_epoch_end(self):
      'Updates indexes after each epoch'
      self.indexes = np.arange(len(self.list_IDs))
      if self.shuffle == True:
          np.random.shuffle(self.indexes)
      
    def __data_generation(self, list_IDs_temp):
      'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
      # Initialization
      t1 = np.empty((self.batch_size, *self.dim,1))
      t2 = np.empty((self.batch_size, *self.dim,1))
      t3 = np.empty((self.batch_size, *self.dim,1))
      t4 = np.empty((self.batch_size, *self.dim,1))
      t5 = np.empty((self.batch_size, *self.dim,1))
      t6 = np.empty((self.batch_size, *self.dim,1))
      #################################################################################
      ty1 = np.empty((self.batch_size, *self.dim,1))
      ty2 = np.empty((self.batch_size, *self.dim,1))
      ty3 = np.empty((self.batch_size, *self.dim,1))
      ty4 = np.empty((self.batch_size, *self.dim,1))
      ty5 = np.empty((self.batch_size, *self.dim,1))
      ty6 = np.empty((self.batch_size, *self.dim,1))      
      

      # Generate data
      for i, ID in enumerate(list_IDs_temp):
          
          t_y = np.load(self.path + '/Y/'+ ID + '.npy')
          t_x = np.load(self.path + '/X/'+ ID + '.npy')
          # Store sample
          t1[i,:,:,0] = t_x[:,:,0]
          t2[i,:,:,0] = t_x[:,:,1]
          t3[i,:,:,0] = t_x[:,:,2]
          t4[i,:,:,0] = t_x[:,:,3]
          t5[i,:,:,0] = t_x[:,:,4]
          t6[i,:,:,0] = t_x[:,:,5]
          
          ty1[i,:,:,0] = t_y[:,:,0]
          ty2[i,:,:,0] = t_y[:,:,1]
          ty3[i,:,:,0] = t_y[:,:,2]
          ty4[i,:,:,0] = t_y[:,:,3]
          ty5[i,:,:,0] = t_y[:,:,4]
          ty6[i,:,:,0] = t_y[:,:,5]
      return [t1,t2,t3,t4,t6,t6],[ty1,ty2,ty3,ty4,ty5,ty6]
    
    def __len__(self):
      'Denotes the number of batches per epoch'
      return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
      'Generate one batch of data'
      # Generate indexes of the batch
      indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
    
      # Find list of IDs
      list_IDs_temp = [self.list_IDs[k] for k in indexes]
    
      # Generate data
      X, y = self.__data_generation(list_IDs_temp)
    
      return X, y