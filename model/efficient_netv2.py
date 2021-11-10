# import the necessary packages
import tensorflow as tf
import keras_efficientnet_v2 as efn
def build_model(HEIGHT,WIDTH,numClasses):
    pretrained = efn.EfficientNetV2S(pretrained="imagenet",num_classes=0,input_shape=[HEIGHT,WIDTH, 3])
    x = pretrained.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation = tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    x =  tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation = tf.keras.layers.LeakyReLU(alpha=0.2))(x)
    outputs = tf.keras.layers.Dense(numClasses,activation="softmax", dtype='float32')(x)
    model = tf.keras.Model(pretrained.input, outputs)
    return model
