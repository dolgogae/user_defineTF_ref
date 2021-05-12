import tensorflow as tf
from tensorflow import keras
import numpy as np

def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

# def my_l1_regularizer(weights):
#     return tf.reduce_sum(tf.abs(0.01 * weights))

class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor=0.01):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}

def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)

layer = keras.layers.Dense(30, activation=my_softplus,
                            kernel_initializer=my_glorot_initializer(0.01),
                            kernel_regularizer=MyL1Regularizer,
                            kernel_constraint=my_positive_weights)

model = keras.models.Sequential()
model = keras.models.load_model("~.h5", custom_objects={"MyL1Regularizer": MyL1Regularizer})

# class화 시켰을 때 얻을 수 있는 이익은 model을 load할 때, 
# 변수(사용자가 입력하고 싶은 값)를 따로 입력할 필요 없음