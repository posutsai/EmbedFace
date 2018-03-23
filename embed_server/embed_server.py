#!/usr/local/bin/python3
import sys
from embed_server import facenet
from embed_server.facenet import crop, prewhiten, flip, to_rgb
from aiohttp import web
import glob
import tensorflow as tf
import numpy as np
import argparse
import math
import os
import pickle
import requests
from server_config import detect_server
import cv2
from tempfile import NamedTemporaryFile
import json
import urllib

parser = argparse.ArgumentParser()
parser.add_argument('--image_size', type=int,
    help='Image size (height, width) in pixels.', default=160)
parser.add_argument('--classifier_filename',
    default='./embed_server/classifier.pkl',
    help='Classifier model file name as a pickle (.pkl) file. ' + 
    'For training this is the output and for classification this is an input.')
args = parser.parse_args()

graph_def = tf.Graph().as_default()
sess = tf.InteractiveSession()
facenet.load_model('./embed_server/model')
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
classifier_filename_exp = os.path.expanduser(args.classifier_filename)
print('loading classifier')
with open(classifier_filename_exp, 'rb') as infile:
    model, class_names = pickle.load(infile)

def img_preprocess(img, do_random_crop, do_random_flip, image_size, do_prewhiten=True):
    images = np.zeros((1, image_size, image_size, 3))
    if img.ndim == 2:
        img = to_rgb(img)
    if do_prewhiten:
        img = prewhiten(img)
    img = crop(img, do_random_crop, image_size)
    img = flip(img, do_random_flip)
    images[0,:,:,:] = img
    return images


def decode_img(content):
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

def detect_face(frame):
    ntf = NamedTemporaryFile(suffix='.jpg')
    cv2.imwrite(ntf.name, frame)
    r = requests.post(detect_server, files={'image': open(ntf.name, 'rb')})
    ntf.close()
    return json.loads(r.text)['bounding_boxes']

def classify(img):
    global images_placeholder
    global embeddings
    global phase_train_placeholder
    global sess
    global args
    global model
    global class_names
    bbs = detect_face(img)
    if len(bbs) is not 1:
        web.json_response({'error': 'it should be only one person on the image'})
    bb = bbs[0]
    cropped = img[ bb[1]:bb[3], bb[0]:bb[2], :]
    scaled = cv2.resize(cropped, (args.image_size, args.image_size))
    images = img_preprocess(scaled, False, False, args.image_size)

    feed_dict = { images_placeholder: images, phase_train_placeholder:False }
    embedding = sess.run(embeddings, feed_dict=feed_dict)

    predictions = model.predict_proba(embedding)
    best_class_indices = np.argmax(predictions, axis=1)
    cls = class_names[best_class_indices[0]]
    prob = predictions[0][best_class_indices[0]]
    return cls, prob

def url2image(url):
    resp = urllib.request.urlopen(url)
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

async def classify_with_file(request):
    data = await request.post()
    img_file = data['image'].file
    img = decode_img(img_file.read())
    cls, prob = classify(img)
    return web.json_response({'class': cls, 'prob': prob})

async def classify_with_url(request):
    j = json.loads(await request.text())
    img = url2image(j['url'])
    cls, prob = classify(img)
    return web.json_response({'class': cls, 'prob': prob})

def run_server():
    app = web.Application()
    app.router.add_post('/image', classify_with_file)
    app.router.add_post('/url', classify_with_url)
    web.run_app(app, host='0.0.0.0', port=8080)

if __name__ == '__main__':
    run_server()
