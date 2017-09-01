import tensorflow as tf

# Prepare to feed input, i.e. feed_dict and placeholders
w1 = tf.placeholder("float", name="w1")
w2 = tf.placeholder("float", name="w2")
b1 = tf.Variable(2.0, name="bias")
feed_dict = {w1: 4, w2: 8}

# Define a test operation that we will restore
w3 = tf.add(w1, w2)
w4 = tf.multiply(w3, b1, name="op_to_restore")
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Create a saver object which will save all the variables
saver = tf.train.Saver()

# Run the operation by feeding input
sess.run(w4, feed_dict)
# Prints 24 which is sum of (w1+w2)*b1

# Now, save the graph
saver.save(sess, 'my_test_model', global_step=1000)


tf.reset_default_graph()

saver = tf.train.import_meta_graph('my_test_model-1000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))

graph = tf.get_default_graph()
w1 = graph.get_tensor_by_name("w1:0")
w2 = graph.get_tensor_by_name("w2:0")
feed_dict = {w1: 13.0, w2: 17.0}

op_to_restore = graph.get_tensor_by_name("op_to_restore:0")

# Add more to the current graph
add_on_op = tf.multiply(op_to_restore, 2)

print(sess.run(add_on_op, feed_dict))
