# import the necessary packages
import tensorflow as tf
import numpy as np
from build_siamese_pairs import *
import config
from utils import list_images
import os
def load_images(input,channels=3,shape=(224,224)):
	imageA = tf.io.read_file(input[0])
	imageA = tf.image.decode_image(imageA, channels=channels)
	imageA = tf.image.resize(imageA,shape) / 255.0
	imageB = tf.io.read_file(input[1])
	imageB = tf.image.decode_image(imageB, channels=channels)
	imageB = tf.image.resize(imageB,shape) / 255.0
	encodedLabel = input[2]
	# return the image and the integer encoded label
	return (imageA,imageB,encodedLabel)

def augment(imageA,imageB,label):
	return (imageA,imageB,label)


#prepare train dataset
train_paths=list_images(config.TRAIN_PATH)
trainlabels = [f.parent.name for f in train_paths]
train_paths=list(map(str, train_paths))
assert len(train_paths) == len(trainlabels)
#prepare val dataset
val_paths=list_images(config.VAL_PATH)
vallabels = [f.parent.name for f in val_paths]
val_paths=list(map(str, val_paths))
assert len(val_paths) == len(vallabels)
#prepare test dataset
test_paths=list_images(config.TEST_PATH)
testlabels = [f.parent.name for f in test_paths]
test_paths=list(map(str, test_paths))
assert len(test_paths) == len(testlabels)


trainX,trainY,dictmap=make_random_pairs(train_paths,trainlabels,labelsmap=True)
valX,valY,_=make_random_pairs(val_paths,vallabels,labelsmap=True)
testX,testY,_=make_random_pairs(test_paths,testlabels,labelsmap=True)
print("[INFO] creating a tf.data input pipeline..")
#build train data pipeline
train_input=np.concatenate((trainX, trainY), axis=1)
trainDS = tf.data.Dataset.from_tensor_slices(train_input)
trainDS = (trainDS
	.shuffle(train_input.shape[0])
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
	.cache()
	.batch(config.BS)
	.prefetch(tf.data.AUTOTUNE)
)

#build val data pipeline
val_input=np.concatenate((valX, valY), axis=1)
valDS = tf.data.Dataset.from_tensor_slices(val_input)
valDS = (valDS
	.shuffle(val_input.shape[0])
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
	.cache()
	.batch(config.BS)
	.prefetch(tf.data.AUTOTUNE)
)

#build test data pipeline
test_input=np.concatenate((testX, testY), axis=1)
testDS = tf.data.Dataset.from_tensor_slices(test_input)
testDS = (testDS
	.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
	.cache()
	.batch(1)
	.prefetch(tf.data.AUTOTUNE)
)


