# import the necessary packages
import tensorflow as tf
import numpy as np
from dataset.build_siamese_pairs import *
import config
from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing
from dataset.utils import list_images
channels=config.channels
shape=config.IMAGE_SHAPE
def load_images(input1,input2,input3):
	imageA = tf.io.read_file(input1)
	imageA = tf.image.decode_image(imageA, channels=3)
	imageA = tf.image.resize(imageA, shape)
	imageB = tf.io.read_file(input2)
	imageB = tf.image.decode_image(imageB, channels=channels)
	imageB = tf.image.resize(imageB,shape)
	encodedLabel = input3
	# return the image and the integer encoded label
	return (imageA,imageB,encodedLabel)


trainAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255),
	preprocessing.RandomFlip("horizontal_and_vertical"),
	preprocessing.RandomZoom(
		height_factor=(-0.05, -0.15),
		width_factor=(-0.05, -0.15)),
	preprocessing.RandomRotation(0.3)
])

testAug = Sequential([
	preprocessing.Rescaling(scale=1.0 / 255)
])

def augment(imageA,imageB,label):
	#do data augmentation
	return (imageA,imageB,label)

def build_dataset():
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


	trainA,trainB,trainY,dictmap=make_random_pairs(train_paths,trainlabels,labelsmap=True)
	valA,valB,valY,_=make_random_pairs(val_paths,vallabels,labelsmap=True)
	testA,testB,testY,_=make_random_pairs(test_paths,testlabels,labelsmap=True)


	print("[INFO] creating a tf.data input pipeline..")

	#build train data pipeline
	tA=tf.data.Dataset.from_tensor_slices(trainA)
	tB=tf.data.Dataset.from_tensor_slices(trainB)
	tY=tf.data.Dataset.from_tensor_slices(trainY)
	trainDS = tf.data.Dataset.zip((tA,tB,tY))
	trainDS = (trainDS
		.shuffle(len(trainA))
		.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
		.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
		.cache()
		.batch(config.BS)
		.prefetch(tf.data.AUTOTUNE)
	)

	#build val data pipeline
	vA=tf.data.Dataset.from_tensor_slices(valA)
	vB=tf.data.Dataset.from_tensor_slices(valB)
	vY=tf.data.Dataset.from_tensor_slices(valY)
	valDS = tf.data.Dataset.zip((vA,vB,vY))
	valDS = (valDS
		.shuffle(len(valA))
		.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
		.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
		.cache()
		.batch(config.BS)
		.prefetch(tf.data.AUTOTUNE)
	)

	#build test data pipeline
	tA=tf.data.Dataset.from_tensor_slices(testA)
	tB=tf.data.Dataset.from_tensor_slices(testB)
	tY=tf.data.Dataset.from_tensor_slices(testY)
	testDS = tf.data.Dataset.zip((tA,tB,tY))
	testDS = (testDS
		.map(load_images, num_parallel_calls=tf.data.AUTOTUNE)
		.cache()
		.batch(1)
		.prefetch(tf.data.AUTOTUNE)
	)
	return trainDS,valDS,testDS,dictmap

