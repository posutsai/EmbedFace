#!/usr/local/bin/python3
import sys
from aiohttp import web
import detect_face as df
import numpy as np
import tensorflow as tf
import requests
import cv2
from tempfile import NamedTemporaryFile
from scipy import misc
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--host', type=str, help='server host for tfserving', default='localhost')
parser.add_argument('--port', type=int, help='server port for tfserving', default=9000)
args = parser.parse_args()

def normalize(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def load_align_img(img, image_size, margin, gpu_memory_fraction):

    minsize = 20 # minimum size of face
    threshold = [ 0.6, 0.7, 0.7 ]  # three steps's threshold
    factor = 0.709 # scale factor

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = df.create_mtcnn(sess, './model/MTCNN')

    img_size = np.asarray(img.shape)[0:2]
    bounding_boxes, _ = df.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) == 0:
        return None, None
    img_list = []
    bbs = []
    for j in range(len(bounding_boxes)):
        det = np.squeeze(bounding_boxes[j,0:4])
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        bbs.append(bb.tolist())
        cropped = img[bb[1]:bb[3],bb[0]:bb[2],:]
        print(cropped.shape)
        # aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        # normalized = normalize(aligned)
        # img_list.append(normalized)
    return bbs

def arrange_predict_response(predres):
    size_tup = tuple(map(lambda e: int(e['size']), predres['outputs']['embeddings']['tensorShape']['dim']))
    return np.reshape(np.array(predres['outputs']['embeddings']['floatVal']), size_tup)

def decode_img(content):
    img_array = np.asarray(bytearray(content), dtype=np.uint8)
    return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

async def process_file(request):
    data = await request.post()
    img_file = data['image'].file
    img = decode_img(img_file.read())
    bounding_boxes = load_align_img(img, 160, 44, 0.3)
    return web.json_response({'bounding_boxes': bounding_boxes})

async def process_array(request):
    data = await request.post()
    img_file = data['image']
    img = decode_img(img_file)

def run_server():
    app = web.Application()
    app.router.add_post('/file', process_file)
    app.router.add_post('/array', process_array)
    web.run_app(app)

if __name__ == '__main__':
    run_server()
