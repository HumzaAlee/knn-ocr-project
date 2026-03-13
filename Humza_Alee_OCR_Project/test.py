import os
import pickle
import numpy as np
from utils.preprocess import preprocess_image


with open('models/knn_model.pkl', 'rb') as f: #load saved knn model
    knn_model = pickle.load(f)

folder_path = input('Enter the path to the folder with test images: ')

x_test = []
file_names = []  


for file in os.listdir(folder_path):
    if file.endswith('.png'):
        image_path = os.path.join(folder_path, file)
        processed = preprocess_image(image_path)
        x_test.append(processed)  #append the processed image to the list
        file_names.append(file)  #store the file names for reference
        


prediction = knn_model.predict(x_test) #predicts labels for the test data

for i in range(len(x_test)):
    final_prediction = prediction[i]
    probabilities = knn_model.predict_proba([x_test[i]])[0] #get the probabilities for each label

    class_index = list(knn_model.classes_).index(final_prediction) #get the index for the prediction
    confidence_score = probabilities[class_index] #get the confidence score for the prediction

    print('File Name: ' + file_names[i] + '\nFinal Predicton:' + final_prediction + '\nConfidence Score: ' + str(int(confidence_score * 100)) + '%' + '\n\n') #prints prediction and confidence score

