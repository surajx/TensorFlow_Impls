"""
This recaps the getting started with TF Core principles tutorial
"""

import tensorflow as tf

sess = tf.Session()

# Addition in TF using constants
a = tf.constant(3.0, dtype=tf.float32)
b = tf.constant(4.0)  # Type is inferred from the value
adder_node = tf.add(a, b)
print("Addition with constants:", sess.run(adder_node))


# Addition in using Placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print("Addition with placeholders:", sess.run(adder_node, {a: 3.0, b: 4.0}))
print("Addition with placeholders:", sess.run(
    adder_node, {a: [3.0, 4.3], b: [2.5, 58.3]}))
