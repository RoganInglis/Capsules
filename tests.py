import tensorflow as tf
import numpy as np
from models import utils


def squash_test():
    test_tensor = np.expand_dims(np.stack([np.zeros([3, 3]), np.ones([3, 3])], axis=2), 0)

    placeholder = tf.placeholder('float', [None, 3, 3, 2])

    squash_out = utils.squash(placeholder, axis=3)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init_op)
    squash_result = sess.run(squash_out, feed_dict={placeholder: test_tensor})


    print("test")


def matmul_test():

    input = tf.expand_dims(tf.constant(np.arange(1, 5, dtype=np.int32), shape=[2, 2]), axis=0)
    weights = tf.constant(np.arange(1, 5, dtype=np.int32), shape=[2, 2])

    matmul_out = tf.matmul(input, weights)

    init_op = tf.global_variables_initializer()

    sess = tf.Session()

    sess.run(init_op)

    np_matmul_out = sess.run(matmul_out)

    print('test')


if __name__ == '__main__':
    #squash_test()
    matmul_test()
