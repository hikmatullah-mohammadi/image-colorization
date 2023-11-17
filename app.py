from flask import Flask, render_template, redirect, request, url_for, session
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import io
import base64
import os

import utils

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY')
app.secret_key = app.config['SECRET_KEY']
# app.debug = False

# Load the trained model
model = tf.keras.models.load_model('./ml-models/img-colorization-model.h5', compile=False)

# Define a route for the home page
@app.route('/')
def home():
    err = session.pop('err', None)
    return render_template('index.html', err=err)

# Define a route for the colorization page
@app.route('/colorize', methods=['POST'])
def colorize():
    try:
      # Get the uploaded grayscale image
      img_file = request.files['bw_img']
      img = cv2.imdecode(np.fromstring(img_file.read(), np.uint8), cv2.IMREAD_COLOR)
      height, width, _ = img.shape
      is_grayscale = utils.is_grayscale(img)
      if is_grayscale:
        L = utils.preprocess_input_image(img)
        # colorize the image
        colorized_img = utils.colorize_image(model, L)[0]
        ##### Get the image ready to be rendered to the html page
        # change the image array to Image object
        colorized_img = Image.fromarray(np.uint8(colorized_img))
        bw_img = Image.fromarray(np.uint8(img))
        # resize back to the image original size
        colorized_img = colorized_img.resize((width, height), Image.ANTIALIAS)
        # Convert the colorized image to a base64 string
        buffered_color = io.BytesIO()
        buffered_bw = io.BytesIO()
        colorized_img.save(buffered_color, format="JPEG")
        bw_img.save(buffered_bw, format='JPEG')

        color_img_str = base64.b64encode(buffered_color.getvalue()).decode('ascii')
        bw_img_str = base64.b64encode(buffered_bw.getvalue()).decode('ascii')
        
        # Render the colorized image on the colorization page
        return render_template('colorize.html', color_img_str=color_img_str, bw_img_str=bw_img_str)
      else:
        # not a grayscale image
        session['err'] = 'Please upload a grayscale image!'
        return redirect(url_for('home'))
    except Exception as err:
      session['err'] = 'Something went wrong. Please try again.'
      print(err)
      return redirect(url_for('home'))

@app.errorhandler(404)
def page_not_found(error):
    return render_template('page_not_found.html')