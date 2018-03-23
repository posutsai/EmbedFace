#!/usr/local/bin/python3
import sys
import facenet
from sklearn.neighbors import KNeighborsClassifier
import glob
import tensorflow as tf
import numpy as np
import argparse
import math
import os
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='./model', help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=90)
parser.add_argument('--classifier_filename',
    default='./classifier.pkl',
    help='Classifier model file name as a pickle (.pkl) file. ' + 
    'For training this is the output and for classification this is an input.')
parser.add_argument('--image_size', type=int,
    help='Image size (height, width) in pixels.', default=160)
parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'], help='Indicates if a new classifier should be trained or a classification ' + 'model should be used for classification', default='CLASSIFY')
parser.add_argument('data_dir', type=str,
    help='Path to the data directory containing aligned LFW face patches.')
parser.add_argument('is_remove', type=bool)
args = parser.parse_args()

def train():
    ## setup encoder
    with tf.Graph().as_default():
        with tf.Session() as sess:
            dataset = facenet.get_dataset('../thumbnails')
            # Check that there are at least one training image per class
            for cls in dataset:
                assert(len(cls.image_paths)>0, 'There must be at least one image for each class in the dataset')
            paths, labels = facenet.get_image_paths_and_labels(dataset)

            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))

            facenet.load_model(args.model_path)

            # Get input and output tensors
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0*nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i*args.batch_size
                end_index = min((i+1)*args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder:images, phase_train_placeholder:False }
                emb_array[start_index:end_index,:] = sess.run(embeddings, feed_dict=feed_dict)
            if (args.mode == 'TRAIN'):
                print('Training classifier')
                model = KNeighborsClassifier(n_neighbors=3, weights='distance')
                model.fit(emb_array, labels)
                # Saving classifier model
                classifier_filename_exp = os.path.expanduser(args.classifier_filename)
                class_names = [ cls.name for cls in dataset]
                print(class_names)
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump((model, class_names), outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

if __name__ == '__main__':
    print(args.is_remove)
    # train()
