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

model.add(Conv2D(64, (2, 2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(image_height, image_width),
    batch_size = batch_size, 
    class_mode = 'categorical'
)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(image_height, image_width), 
    batch_size = batch_size, 
    class_mode= 'categorical'
)

model.fit(
    train_generator, 
    steps_per_epoch = train_samples // batch_size, epochs = epochs, validation_data = validation_generator, 
    validation_steps=validation_samples // batch_size
)

#model.save_weights('saved_model.h5')
model.save("my_model")