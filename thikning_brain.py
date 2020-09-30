from keras.preprocessing.image import ImageDataGenerator 
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense 
from keras import backend as K 

image_height, image_width = 224, 224 

train_data_dir = 'dataset/train_data'
validation_data_dir = 'dataset/validation_data'
train_samples = 11500 
validation_samples = 1000
epochs = 10
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, image_width, image_height)
else: 
    input_shape = (image_width, image_height, 3)

model = Sequential()
model.add(Conv2D(32, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))