
# Silensing import warnings
import warnings
warnings.filterwarnings('ignore')
import argparse
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd 
import json
import sys
from PIL import Image
import os.path
import logging


def get_image_and_model():
    try:
        return str(sys.argv[1]), str(sys.argv[2]) #first return is the image path, second is the model name
    except Exception as error:
        print(error)
        sys.exit()
        
def open_entered_options():
    # Reading category_names and top_k
    commands = parser.parse_args(sys.argv[3:])
    category_names = commands.category_names
    top_k = commands.top_k
    # Opening category_names json file
    try:
        with open(category_names, 'r') as f:
            category_names = json.load(f)
    except Exception as error:
        print(error)
        sys.exit()
    return category_names, top_k

def open_model(model_name):
    return tf.keras.models.load_model(model_name
                    , custom_objects = {'KerasLayer': hub.KerasLayer})

def process_image(image):
    x_float32 = tf.convert_to_tensor(image, tf.float32)
    processed_image = tf.image.resize(x_float32, (224, 224))
    return processed_image.numpy() / 255

def predict(image_path, model, top_k):
    im = Image.open(image_path)
    image = np.asarray(im)
    image = process_image(image)
    image = np.expand_dims(image, axis = 0)
    prediction = model.predict(image)
    prediction_dict = {str(v): k for v, k in enumerate(prediction[0], 1)}
    sorted_prediction = dict(sorted(prediction_dict.items(), key=lambda item: item[1], reverse = True))
    return list(sorted_prediction.values())[:top_k], list(sorted_prediction)[:top_k]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--top_k'
                        , action = "store"
                        , type = int
                        , default = 5
                        , help = 'Number of probabilties to show'
                       )

    parser.add_argument('--category_names'
                        , action = "store"
                        , default = 'label_map.json'
                        , help = 'File name that has the label names'
                       )
    image_path, model_name = get_image_and_model()
    category_names, top_k = open_entered_options()
    reloaded_model = open_model(model_name)
    prob, classes = predict(image_path, reloaded_model, top_k)
    labels = []
    for label_num in classes:
        labels.append(category_names[label_num])
    print('Probabilities: {}'.format(prob))
    print('Classes: {}'.format(labels))