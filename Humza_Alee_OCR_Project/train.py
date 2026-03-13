from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from utils.preprocess import load_dataset
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle
import warnings
import pandas as pd

warnings.filterwarnings('ignore', category=UserWarning) #ignores user warnings from sklearn


dataset_path = 'image_files'  #load the dataset from my image folder
dataset = load_dataset(dataset_path)


x = np.array([item[0] for item in dataset]) #seperates image data and labels from the list of tuples
y = np.array([item[1] for item in dataset])  

knn_model = KNeighborsClassifier(n_neighbors = 151, weights ='distance') #creates a knn model

knn_model.fit(x, y)


with open('models/knn_model.pkl', 'wb') as f:
    pickle.dump(knn_model, f)  #save to file

print('Model trained and saved successfully.')


