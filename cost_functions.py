import scipy
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from nst_utils import *
import PIL


def compute_content_cost(a_C, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    
    a_C_unrolled = tf.reshape(a_C, (m, n_H * n_W, n_C))
    a_G_unrolled = tf.reshape(a_G, (m, n_H * n_W, n_C))
    
    J_content = tf.multiply(1/(4*n_H*n_W*n_C), tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled))))
   
    return J_content




#creating gram matrix
def gram_matrix(A):
    
    GA = tf.matmul(A, tf.transpose(A))
    return GA



def compute_layer_style_cost(a_S, a_G):
    
    m, n_H, n_W, n_C = a_S.get_shape().as_list()
    
    a_S = tf.transpose(tf.reshape(a_S, (n_H * n_W, n_C)))
    a_G = tf.transpose(tf.reshape(a_G, (n_H * n_W, n_C)))

    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    
    J_style_layer =  tf.reduce_sum(tf.square(tf.subtract(GS, GG)))/4.0/(n_C*n_H*n_W)**2.0
    return J_style_layer



STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]



def compute_style_cost(model, STYLE_LAYERS):


    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        out = model[layer_name]

        a_S = sess.run(out)

        a_G = out
        
        J_style_layer = compute_layer_style_cost(a_S, a_G)
        J_style += coeff * J_style_layer

    return J_style


# GRADED FUNCTION: total_cost

def total_cost(J_content, J_style, alpha = 10, beta = 40):

    J = alpha*J_content+ beta*J_style
    
    return J








