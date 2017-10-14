"""
The linear model implementation from the getting started tutorial
"""

import tensorflow as tf

with tf.Session() as sess:
    W = tf.get_variable("W", initializer=tf.constant([0.3]), trainable=True)
    b = tf.get_variable("b", initializer=tf.constant([-0.3]))

    X = tf.placeholder(tf.float32)

    # Note that here W is a scalar
    linear_model = W * X + b  # \hat{y} = w^Tx + b
    # init the variables in the session
    sess.run(tf.global_variables_initializer())
    # Run the untrained linear model.
    print(sess.run(linear_model, {X: [1, 2, 3, 4]}))

    # Known Labels
    y = tf.placeholder(tf.float32)
    # Calculate sum of squared error
    loss = tf.reduce_sum(tf.square(linear_model - y))
    print("Loss:", sess.run(loss, {X: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

    # Changing value of initialized variable.
    # "changeW" is an assign node, which when run assigns the Variable "W" with
    # value [-1.]. Similar for changeb.
    changeW = tf.assign(W, [-1.])
    changeb = tf.assign(b, [1.])

    sess.run([changeW, changeb])
    print("Loss with new W,b:", sess.run(
        loss, {X: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
