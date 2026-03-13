import os
from PIL import Image
import numpy as np

image_size = (28, 28) 


def preprocess_image(image_path):
    img = Image.open(image_path) #uses pillow to open the image file
    img = img.convert('L') 
    img = img.resize(image_size) 
    img_array = np.array(img) / 255 #converts image to numpy array and normalise pixel values to avoid bias to ones that are numerically larger
    return img_array.flatten() #flattens the array to a 1D array so it can be used as an input to the model
    
def load_dataset(dataset_path):
    dataset = []

    for file in os.listdir(dataset_path):
        if file.endswith('.png'):
            image_path = os.path.join(dataset_path, file) 

            label_unicode = file.split('_')[-1].replace('.png','')  #Extract the unicode for the label from filename
            label = chr(int(label_unicode))

        
            processed = preprocess_image(image_path)

            dataset.append((processed, label))  #append the processed image and its label
    return dataset  #return the list of processed images and their labels






