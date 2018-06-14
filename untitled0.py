import keras.backend as k
import keras
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

path = 'data\test'

image_data = ImageDataGenerator().flow_from_directory(path, batch_size=3, classes=['cat', 'dogs'])










