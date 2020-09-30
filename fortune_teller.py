from keras.models import load_model
from keras.preprocessing import image 

model = load_model('saved_model.h5')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
