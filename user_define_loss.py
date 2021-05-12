import tensorflow as tf
from tensorflow import keras
import numpy as np

# def create_huber(threshold=1.0):
#     def huber_fn(y_true, y_pred):
#         error = y_true - y_pred
#         is_small_error = tf.abs(error) < threshold
#         squared_loss = tf.square(error) / 2
#         linear_loss = threshold * tf.abs(error) - threshold**2 / 2
#         return tf.where(is_small_error, squared_loss, linear_loss)
#     return huber_fn

# model = keras.models.Sequential()
# model.compile(loss=create_huber(2.0), optimizer='nadam')

# model = keras.models.load_model("~.h5", custom_objects={"huber_fn":create_huber(2.0)})

class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


model = keras.models.Sequential()
model.compile(loss=HuberLoss(2.0), optimizer='nadam')

model = keras.models.load_model("~.h5", custom_objects={"HuberLoss":HuberLoss})

#===============================================
#지표