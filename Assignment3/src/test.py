import numpy as np 
import pandas as pd
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt
import cv2
from time import time
import sys,os
# Import TF and TF Hub libraries.
import tensorflow as tf
np.random.seed(46)
import tensorflow, gc
from keras.preprocessing.image import ImageDataGenerator
import keras
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow import keras as keras
tf.random.set_seed(46)

model_path=sys.argv[1]
testfilename=sys.argv[2]
submissionfilename=sys.argv[3]

class DataGenerator(tensorflow.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data,train=True,val=False, batch_size=32, n_channels=3,
                 n_classes=19, shuffle=True):
        'Initialization'
        #self.dim = dim
        self.train=train
        self.val=val
        self.batch_size = batch_size
        if(self.train):
            self.labels = data.iloc[:,1].to_numpy()
            self.list_IDs = data.iloc[:,0].to_numpy()
        else:
            self.labels = data.to_numpy().squeeze()
            self.list_IDs = data.to_numpy().squeeze()
        self.total=self.labels.shape[0] 
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.augmentor = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.15,
            fill_mode='nearest',
            horizontal_flip=True,
        )
        #Image.fromarray(numpy_image.astype('uint8'), 'RGB')
    def __len__(self):
        'Denotes the number of batches per epoch'
        if (self.train==False):
            return int(np.floor(len(self.list_IDs) / self.batch_size))+1
        else:
            return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
    
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
       
        # Find list of IDs
        image_names = [self.list_IDs[k] for k in indexes]
        if(self.train):
            y=np.array([self.labels[i] for i in indexes],dtype=int)
        # Generate data
        X = self.__data_generation(image_names)
        if(self.train):
            return X, tensorflow.keras.utils.to_categorical(y, num_classes=self.n_classes)
        else:
            return X,np.array(image_names)

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        gc.collect()
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        # os.chdir("/kaggle/input/col341-a3")
        dir_path=os.path.dirname(testfilename)
        X = np.empty((self.batch_size, 256,256, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(dir_path + ID)[-256:,128:128+256,:]
        if(self.train==True and self.val==False):
            X = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)[0]
        gc.collect()
        # os.chdir("/kaggle/working")
        return X

private_test_data=pd.read_csv(testfilename)
d={0:'Ardhachakrasana', 1:'Garudasana', 2:'Gorakshasana', 3:'Katichakrasana',
       4:'Natarajasana', 5:'Natavarasana', 6:'Naukasana', 7:'Padahastasana',
       8:'ParivrittaTrikonasana', 9:'Pranamasana', 10:'Santolanasana',11: 'Still',
       12:'Tadasana', 13:'Trikonasana', 14:'TriyakTadasana', 15:'Tuladandasana',
       16:'Utkatasana', 17:'Virabhadrasana',18: 'Vrikshasana'}

private_test=DataGenerator(private_test_data,False,False,39,shuffle=False)

model1=tf.keras.models.load_model(model_path+'/effb4')
model2=tf.keras.models.load_model(model_path +'/resnet50v2')
model3=tf.keras.models.load_model(model_path +'/xception')
model4=tf.keras.models.load_model(model_path +'/effb5')

preds1=model1.predict_generator(private_test)
preds2=model2.predict_generator(private_test)
preds3=model3.predict_generator(private_test)
preds4=model4.predict_generator(private_test)

preds1=preds1[:private_test_data.shape[0]]
preds2=preds2[:private_test_data.shape[0]]
preds3=preds3[:private_test_data.shape[0]]
preds4=preds4[:private_test_data.shape[0]]

preds=preds1*6.3+preds2*6.3+preds3*6.3+preds4*6
preds=np.argmax(preds,axis=1)

submission=[]
names=private_test_data['name'].to_numpy().squeeze()
for i in range(preds.shape[0]):
    submission.append([names[i],d[preds[i]]])

submission=np.array(submission)
submission = pd.DataFrame(submission, columns = ['name','category']).iloc[:-1]
submission.to_csv(submissionfilename,index=False)


