import flask
from flask import Flask, render_template,request
import imageio as imgio
import numpy as np
import tensorflow.keras.models
import os
import jsonpickle
import cv2
from werkzeug.utils import secure_filename

from loadmodel import * 

app = Flask(__name__)
#global model
#model = init()

#models only used for determining what model to use
global digit_letter_clsfier, letter_case_clsfier
#models used for determining what a given input is
global digit_model, letter_lower_model, letter_upper_model
global by_class_model

digit_letter_clsfier, letter_case_clsfier = init_clsfier()
digit_model, letter_lower_model, letter_upper_model = init_models()
by_class_model = load_model('EMNIST-byclass-epoch20-87p')


# image has 4 bytes data for each pixel? -> 4 * 28 * 28 = 3136?
# but each foramt has different way of compressing data
#app.config['MAX_CONTENT_LENGTH'] =  10 * 1024  #to prevent large images come in

@app.route('/')
def index():
	return render_template("index.html")

# shows the list of files saved in /user_inputs folder
@app.route('/dir/',methods=['GET','POST'])
def dir():
    out_str = ''
    for file in os.listdir("user_inputs"):
        out_str = out_str + str(file) + ' / '

    response = {'list': out_str}
    response_pickled = jsonpickle.encode(response)
    return flask.Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/test/',methods=['GET','POST'])
def test():
    #with graph.as_default():
    img1 = imgio.imread('static/1.png',pilmode='L')
    img5 = imgio.imread('static/5.png',pilmode='L')
    img7 = imgio.imread('static/7.png',pilmode='L')
    img8 = imgio.imread('static/8.png',pilmode='L')

    flatten = get_flatten([img1, img5, img7, img8])
    np_flatten = np.array(flatten, np.float32)

    prediction = digit_model.predict(np_flatten)
    out_str = ''
    for p in prediction:
        l = list(p)
        out_str = out_str + str(l.index(max(l))) + '  '

    return out_str

@app.route('/byclass/',methods=['GET','POST'])
def byclass():
    imagefile = request.files.get('file', '')

    # if want to reduce disk IO, can directly read into cv2, but not tested much
    #test_img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #print(test_img)
    #cv2.imwrite('test.png', test_img)

    # converting directly into numpy array not working.. so
    # save the file and then read in the saved file
    filename = secure_filename(imagefile.filename)
    save_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(save_path, 'user_inputs') #concat the folder name
    save_path = os.path.join(save_path, filename)
    imagefile.save(save_path)

    img = cv2.imread('user_inputs/' + filename, 0)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.resize(255 - img, (28, 28))
    #ret, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #thresh, result = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #img_result3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    
    fltn = img.flatten() / 255.0
    img_arr = np.array(fltn).reshape(1,28,28,1).astype(np.float32)
    #img_arr = transpose(img_arr)
    
    prediction = by_class_model.predict(img_arr)
    result = get_prediction_category(prediction)
    #print(result)
    #print(chr(ord('A') + result))
    
    if (result < 10):
        result = str(result)
    elif (result < 36):
        result = chr(ord('A') + result - 10)
    else:
        result = chr(ord('a') + result - 36)
    

    # remove the saved image.
    os.remove(save_path)
    
    response = {'result': result}
    response_pickled = jsonpickle.encode(response)

    return flask.Response(response=response_pickled, status=200, mimetype="application/json")


@app.route('/lettertest/',methods=['GET','POST'])
def lettertest():
    imagefile = request.files.get('file', '')

    # if want to reduce disk IO, can directly read into cv2, but not tested much
    #test_img = cv2.imdecode(np.fromstring(request.files['file'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #print(test_img)
    #cv2.imwrite('test.png', test_img)

    # converting directly into numpy array not working.. so
    # save the file and then read in the saved file
    filename = secure_filename(imagefile.filename)
    save_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(save_path, 'user_inputs') #concat the folder name
    save_path = os.path.join(save_path, filename)
    imagefile.save(save_path)

    img = cv2.imread('user_inputs/' + filename, 0)
    img = cv2.GaussianBlur(img,(5,5),0)
    img = cv2.resize(255 - img, (28, 28))
    #ret, result = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    #thresh, result = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #img_result3 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5)
    
    fltn = img.flatten() / 255.0
    img_arr = np.array(fltn).reshape(1,28,28,1).astype(np.float32)
    img_arr = transpose(img_arr)
    
    prediction = letter_upper_model.predict(img_arr)
    result = get_prediction_category(prediction)
    #print(result)
    #print(chr(ord('A') + result))
    result = chr(ord('A') + result)
    

    # remove the saved image.
    os.remove(save_path)
    
    response = {'result': result}
    response_pickled = jsonpickle.encode(response)

    return flask.Response(response=response_pickled, status=200, mimetype="application/json")

@app.route('/predict/',methods=['GET','POST'])
def predict():
    imagefile = request.files.get('file', '')

    # converting directly into numpy array not working.. so
    # save the file and then read in the saved file
    filename = secure_filename(imagefile.filename)
    save_path = os.path.dirname(os.path.realpath(__file__))
    save_path = os.path.join(save_path, 'user_inputs') #concat the folder name
    save_path = os.path.join(save_path, filename)
    imagefile.save(save_path)

    # read in the saved file, and flatten
    img = imgio.imread('user_inputs/' + filename ,pilmode='L')
    flatten = get_flatten([img])
    np_flatten = np.array(flatten, np.float32)

    prediction = digit_model.predict(np_flatten)
    out_str = ''
    for p in prediction:
        l = list(p)
        out_str = out_str + str(l.index(max(l))) + '  '
    
    # remove the saved image.
    os.remove(save_path)
    
    response = {'result': out_str}
    response_pickled = jsonpickle.encode(response)

    return flask.Response(response=response_pickled, status=200, mimetype="application/json")


# input == numpy array shape of 28x28
def get_flatten(img_list):
    flatten = []
    for img in img_list:
        temp = []
        for i in range(0,28):
            for j in range(0,28):
                temp.append(img[i][j])
        flatten.append(temp)
    return flatten

# get result from prediction numpy array (assuming shape (1,n))
def get_prediction_category(predict_arr):
    l = list(predict_arr[0])
    return l.index(max(l))

def predict_digit(img):
    flatten = get_flatten([img])
    np_flatten = np.array(flatten, np.float32)

    prediction = digit_model.predict(np_flatten)
    return get_prediction_category(prediction)

def transpose(img_arr):
    img_arr = img_arr.reshape(28,28).transpose().reshape(1,28,28,1)
    return img_arr
#def predict_letter(img):


if __name__ == "__main__":
	port = int(os.environ.get('PORT', 8080))
	app.run(host='0.0.0.0', port=port)