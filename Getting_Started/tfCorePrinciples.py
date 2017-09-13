"""
This recaps the getting started with TF Core principles tutorial
"""

import tensorflow as tf

sess = tf.Session()

# Adding Constants.
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)
print("Adding Constants:", sess.run(tf.add(a, b)))

# Creating placeholder nodes that can be fed values.
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
add = a + b  # + operator overriden to tf.add
print("Adding Dynamic values:", sess.run(add, {a: 3, b: 4}))
