# USAGE
# python reading_from_disk.py --dataset path_to_folder

# import the necessary packages
from tensorflow.data import AUTOTUNE
from imutils import paths
import tensorflow as tf
import numpy as np
import argparse
import os

def load_directoryimage(imagePath,channels=3,shape=(224,224)):
	# read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_image(image, channels=channels)
	image = tf.image.resize(image,shape) / 255.0
	# grab the label and encode it
	label = tf.strings.split(imagePath, os.path.sep)[-2]
	oneHot = label == classNames
	encodedLabel = tf.argmax(oneHot)

	# return the image and the integer encoded label
	return (image, encodedLabel)

def load_csvimage(imagePath,df,channels=3,shape=(224,224)):
	# read the image from disk, decode it, resize it, and scale the
	# pixels intensities to the range [0, 1]
	image = tf.io.read_file(imagePath)
	image = tf.image.decode_image(image, channels=channels)
	image = tf.image.resize(image,shape) / 255.0
	# grab the label and encode it
	label = df[imagePath][]
	oneHot = label == classNames
	encodedLabel = tf.argmax(oneHot)

	# return the image and the integer encoded label
	return (image, encodedLabel)
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
	my_parser = argparse.ArgumentParser(description='Load dataset from Memory Using tf.data api')
	# Add the arguments
    my_parser.add_argument('--dataset', required=True,metavar='PATH',help='path to the dataset')
	my_parser.add_argument('--display',metavar='PATH',default="siamese_pairs.png",type=str,
                       						help='path to save image')
    # initialize batch size and number of steps
    BS = 64
    NUM_STEPS = 1000

    # grab the list of images in our dataset directory and grab all
    # unique class names
    print("[INFO] loading image paths...")
    imagePaths = list(paths.list_images(args["dataset"]))
    classNames = np.array(sorted(os.listdir(args["dataset"])))

    # build the dataset and data input pipeline
    print("[INFO] creating a tf.data input pipeline..")
    dataset = tf.data.Dataset.from_tensor_slices(imagePaths)
    dataset = (dataset
        .shuffle(1024)
        .map(load_images, num_parallel_calls=AUTOTUNE)
        .cache()
        .repeat()
        .batch(BS)
        .prefetch(AUTOTUNE)
    )



    # create a dataset iterator, benchmark the tf.data pipeline, and
    # display the number of data points generated, along with the time
    # taken
    datasetGen = iter(dataset)