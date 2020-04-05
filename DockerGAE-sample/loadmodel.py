import numpy as np
import tensorflow.keras.models
from tensorflow.keras.models import model_from_json
import tensorflow as tf


def init(): 
    json_file = open('784model.json','r')
    model_json = json_file.read()
    json_file.close()

    model = tensorflow.keras.models.model_from_json(model_json)
    model.load_weights("784model.h5")

    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    #graph = tf.get_default_graph()

    return model, graph

# initialize models only used for determining what model to use
def init_clsfier():
    digit_letter_clsfier = load_model('Digit-Letter-Classifier-epoch10-93p')
    letter_case_clsfier = load_model('EMNIST-CaseClassifier-epoch14-89p')

    return digit_letter_clsfier, letter_case_clsfier

# initialize models used for determining what a given input is
def init_models():
    digit_model = load_model('784model')
    letter_lower_model = load_model('EMNIST-lower-epoch8')
    letter_upper_model = load_model('EMNIST-upper-epoch14-98p')

    return digit_model, letter_lower_model, letter_upper_model

def load_model(model_name, loss='categorical_crossentropy', optimizer='adam'):
    json_file = open('datasets/' + model_name + '.json','r')
    model_json = json_file.read()
    json_file.close()

    model = tensorflow.keras.models.model_from_json(model_json)
    model.load_weights('datasets/' + model_name + '.h5')

    model.compile(loss=loss,optimizer=optimizer,metrics=['accuracy'])
    #graph = tf.get_default_graph()

    return model