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