from keras.models import load_model
from keras.preprocessing import image 

model = load_model('my_model')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(model)