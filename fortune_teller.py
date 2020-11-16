import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from keras.preprocessing import image 
import numpy as np 

model = load_model('my_model')

test_image = image.load_img('dataset/test_data/not-food/photos-167.jpg', target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)

result = model.predict(test_image)
#print(result)

if result[0][0] // 1 == 0: 
    print("Food")
else:
    print("Not Food")