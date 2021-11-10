#import necessary modules
from tensorflow.python.eager.context import num_gpus
from dataset import load_dataset as ld
from model import efficient_netv2 as efv2
import config
import tensorflow as tf
import plots
from pathlib import Path
import distances

#build dataset
print("[INFO] preparing positive and negative pairs...")
trainDS,valDS,testDS,dictmap=ld.build_dataset()
config.numClasses=len(dictmap)

print("[INFO] building siamese network...")
imgA = tf.keras.layers.Input(shape=config.IMG_SHAPE)
imgB = tf.keras.layers.Input(shape=config.IMG_SHAPE)
featureExtractor = efv2.build_model(config.IMG_SHAPE[0],config.IMG_SHAPE[1],num_classes=config.numClasses)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

# finally, construct the siamese network
distance = tf.keras.layers.Lambda(distances.euclidean_distance)([featsA, featsB])
outputs = tf.keras.layers.Dense(1, activation="sigmoid")(distance)
model = tf.keras.model.Model(inputs=[imgA, imgB], outputs=outputs)
