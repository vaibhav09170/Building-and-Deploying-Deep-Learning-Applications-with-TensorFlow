import os
#import tensorflow as tf

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Turn off TensorFlow warning Messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

X = tf.placeholder(tf.float32, name = 'X')
Y = tf.placeholder(tf.float32, name = 'Y')

addition = tf.add(X,Y, name = 'addition')

with tf.Session() as session:
    result = session.run(addition, feed_dict = {X: [1,2,10], Y: [4,2,10]})
    print(result)