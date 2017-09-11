"""
The default program to make sure that the tensorflow installation is
working fine.
"""
import tensorflow as tf

hello = tf.constant('Hello World!')
sess = tf.Session()
print(sess.run(hello))
