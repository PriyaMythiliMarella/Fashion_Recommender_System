import pickle
import tensorflow
import numpy as np
from numpy.linalg import norm
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from sklearn.neighbors import KNeighborsClassifier

import cv2

# Load feature_list and filenames
feature_list = np.array(pickle.load(open('embeddings.pkl', 'rb')))
filenames = pickle.load(open('filenames.pkl', 'rb'))

# Load labels (replace 'your_labels.pkl' with the actual filename)
# labels = pickle.load(open('your_labels.pkl', 'rb'))

model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tensorflow.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])

img = image.load_img('sample/saree.jpg', target_size=(224, 224))
img_array = image.img_to_array(img)
expanded_img_array = np.expand_dims(img_array, axis=0)
preprocessed_img = preprocess_input(expanded_img_array)
result = model.predict(preprocessed_img).flatten()
normalized_result = result / norm(result)

# Create KNeighborsClassifier with specified parameters
neighbors = KNeighborsClassifier(n_neighbors=6, algorithm='brute', metric='euclidean')

# Fit the model with both features and labels
neighbors.fit(feature_list,filenames)

# Find k-nearest neighbors for the new data point
distances, indices = neighbors.kneighbors([normalized_result])

print(indices)

# Display the nearest neighbor images
for file in indices[0][1:6]:
    temp_img = cv2.imread(filenames[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
