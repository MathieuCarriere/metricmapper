from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
from os import listdir
from os.path import isfile, join
import os

path="/home/mathieu/Documents/data/natural_images/"
files = []
for r, d, f in os.walk(path):
	for file in f:
		if ".jpg" in file:
			files.append(os.path.join(r, file))
#print(len(files))

model = ResNet50(weights='imagenet')
P = []
for f in files:	
	print(f)
	img = image.load_img(f, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = model.predict(x)
	#print(preds)
	#print(decode_predictions(preds, top=3)[0])
	P.append(preds)
P = np.vstack(P)
print(P.shape)
np.save("ResNet50_ImageNet_Isini", P)
