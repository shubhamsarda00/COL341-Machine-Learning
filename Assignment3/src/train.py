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
import multiprocessing


trainfilename=sys.argv[1]
model_path=sys.argv[2]

train_data=pd.read_csv(trainfilename)
train_data.head()
train_data['category']=train_data["category"].astype('category').cat.codes
train_data=train_data.sample(frac=1.,random_state=46)
train_data,test_data=train_data.iloc[:29000],train_data.iloc[29000:]

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
        dir_path=os.path.dirname(trainfilename)
        X = np.empty((self.batch_size, 256,256, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(dir_path + ID)[-256:,128:128+256,:]
        if(self.train==True and self.val==False):
            X = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)[0]
#         X=np.array(transforms.Compose([transforms.ToPILImage(),transforms.RandomPerspective()])(X))
        gc.collect()
        # os.chdir("/kaggle/working")
        return X

def train():
    training_generator = DataGenerator(train_data,True,False,24,3,19,True)
    validation_generator = DataGenerator(test_data,True,True,24,3,19,True)


    base_model = tf.keras.applications.EfficientNetB5(
        include_top=False,
        weights="imagenet",
        classes=19,
    )

    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=True)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Dense(1024,activation='relu')(x)
    #x = keras.layers.Dense(128,activation='relu')(x)
    outputs = keras.layers.Dense(19,activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    gc.collect()


    def scheduler(epoch, lr):
        return lr*((.9)**(epoch-1))
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
        monitor="val_acc",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        options=None,
    )
    #tf.config.get_visible_devices()
    lrs = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizers.Adam(learning_rate=6e-4),
              loss='categorical_crossentropy',metrics=["acc"])
    # Train model on dataset
    model.fit_generator(epochs=5,generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,callbacks=[lrs],
                        workers=6)

    model.save(model_path+ "/effb5")

    gc.collect()

p = multiprocessing.Process(target=train)
p.start()
p.join()



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
import multiprocessing
tf.random.set_seed(46)
import random
random.seed(6)


trainfilename=sys.argv[1]
model_path=sys.argv[2]

train_data=pd.read_csv(trainfilename)
train_data.head()
train_data['category']=train_data["category"].astype('category').cat.codes
train_data=train_data.sample(frac=1.,random_state=46)
train_data,test_data=train_data.iloc[:29000],train_data.iloc[29000:]

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
        dir_path=os.path.dirname(trainfilename)
        X = np.empty((self.batch_size, 256,256, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(dir_path + ID)[-256:,128:128+256,:]
        if(self.train==True and self.val==False):
            X = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)[0]
        gc.collect()
        # os.chdir("/kaggle/working")
        return X

def train():
    training_generator = DataGenerator(train_data,True,False,40,3,19,True)
    validation_generator = DataGenerator(test_data,True,True,40,3,19,True)

    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        classes=19,
    )

    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=True)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Dense(1024,activation='relu')(x)
    #x = keras.layers.Dense(128,activation='relu')(x)
    outputs = keras.layers.Dense(19,activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    gc.collect()

    def scheduler(epoch, lr):
        return lr*((.95)**(epoch-1))
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
        monitor="val_acc",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        options=None,
    )
    #tf.config.get_visible_devices()
    lrs = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizers.Adam(learning_rate=5.5e-4),
              loss='categorical_crossentropy',metrics=["acc"])
    # Train model on dataset
    model.fit_generator(epochs=5,generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,callbacks=[lrs],
                        workers=6)
    
    model.save(model_path+ "/xception")

    gc.collect()

p = multiprocessing.Process(target=train)
p.start()
p.join()


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
import multiprocessing

trainfilename=sys.argv[1]
model_path=sys.argv[2]

train_data=pd.read_csv(trainfilename)
train_data.head()
train_data['category']=train_data["category"].astype('category').cat.codes
train_data=train_data.sample(frac=1.,random_state=46)
train_data,test_data=train_data.iloc[:29000],train_data.iloc[29000:]

import tensorflow, gc
from keras.preprocessing.image import ImageDataGenerator
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
        dir_path=os.path.dirname(trainfilename)
        X = np.empty((self.batch_size, 256,256, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(dir_path + ID)[-256:,128:128+256,:]
        if(self.train==True and self.val==False):
            X = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)[0]
        gc.collect()
        # os.chdir("/kaggle/working")
        return X

def train():
    training_generator = DataGenerator(train_data,True,False,40,3,19,True)
    validation_generator = DataGenerator(test_data,True,True,40,3,19,True)


    base_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights="imagenet",
        classes=19,
    )

    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=True)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Dense(1024,activation='relu')(x)
    #x = keras.layers.Dense(128,activation='relu')(x)
    outputs = keras.layers.Dense(19,activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    gc.collect()

    def scheduler(epoch, lr):
        return lr*((.95)**(epoch-1))
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
        monitor="val_acc",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        options=None,
    )
    #tf.config.get_visible_devices()
    lrs = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
              loss="categorical_crossentropy",metrics=["acc"])

    # Train model on dataset
    model.fit_generator(epochs=5,generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,callbacks=[lrs],
                        workers=6)

    model.save(model_path+ "/resnet50v2")

    gc.collect()

p = multiprocessing.Process(target=train)
p.start()
p.join()

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
import multiprocessing
tf.random.set_seed(46)
import random
random.seed(6)


trainfilename=sys.argv[1]
model_path=sys.argv[2]

train_data=pd.read_csv(trainfilename)
train_data.head()
train_data['category']=train_data["category"].astype('category').cat.codes
train_data=train_data.sample(frac=1.,random_state=46)
train_data,test_data=train_data.iloc[:29000],train_data.iloc[29000:]

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
        dir_path=os.path.dirname(trainfilename)
        X = np.empty((self.batch_size, 256,256, self.n_channels))
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = cv2.imread(dir_path + ID)[-256:,128:128+256,:]
        if(self.train==True and self.val==False):
            X = self.augmentor.flow(X, batch_size=self.batch_size, shuffle=False)[0]
        gc.collect()
        # os.chdir("/kaggle/working")
        return X

def train():
    training_generator = DataGenerator(train_data,True,False,40,3,19,True)
    validation_generator = DataGenerator(test_data,True,True,40,3,19,True)


    base_model = tf.keras.applications.EfficientNetB4(
        include_top=False,
        weights="imagenet",
        classes=19,
    )

    inputs = keras.Input(shape=(256, 256, 3))

    x = base_model(inputs, training=True)
    # x = keras.layers.Flatten()(x)
    x = keras.layers.GlobalMaxPool2D()(x)
    x = keras.layers.Dense(1024,activation='relu')(x)
    #x = keras.layers.Dense(128,activation='relu')(x)
    outputs = keras.layers.Dense(19,activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    gc.collect()

    def scheduler(epoch, lr):
        return lr*((.95)**(epoch-1))
    modelcheckpoint=tf.keras.callbacks.ModelCheckpoint(
        "model.h5",
        monitor="val_acc",
        verbose=0,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        save_freq="epoch",
        options=None,
    )
    #tf.config.get_visible_devices()
    lrs = keras.callbacks.LearningRateScheduler(scheduler)
    model.compile(optimizer=optimizers.Adam(learning_rate=5e-4),
              loss='categorical_crossentropy',metrics=["acc"])

    # Train model on dataset
    model.fit_generator(epochs=5,generator=training_generator,
                        validation_data=validation_generator,
                        use_multiprocessing=True,callbacks=[lrs],
                        workers=6)

    model.save(model_path+ "/effb4")

    gc.collect()

p = multiprocessing.Process(target=train)
p.start()
p.join()
