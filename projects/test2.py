from keras import backend as K
import tensorflow as tf
import numpy as np
# b = K.random_uniform_variable(shape=(3, 4), low=0, high=1) # Uniform distribution
# c = K.random_normal_variable(shape=(3, 4), mean=0, scale=1) # Gaussian distribution
# d = K.random_normal_variable(shape=(3, 4), mean=0, scale=1)

# # Tensor Arithmetic
# a = b + c * K.abs(d)
# c = K.dot(a, K.transpose(b))
# a = K.sum(b, axis=1)
# a = K.softmax(b)
# a = K.concatenate([b, c], axis=-1)
# # print(a)

r = np.array([1, 2, 3])
a = tf.convert_to_tensor(r, dtype = tf.float32)
c = 1-a
print(c)
print(type(a), type(c))