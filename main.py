import numpy as np
import tensorflow as tf
import imageio
import glob, os
import matplotlib.pyplot as plt

omni = np.load('omniglot_data.npy')
train = omni[1200:1600]

images_per_class = 20
number_of_classes = 400

use_images_per_class = 5
use_augumented_images_per_class = 5

labels_train = []
data_train = []
labels_test = []
data_test = []

i = 0
for im_path in glob.glob("aug/*.png"):
	im = imageio.imread(im_path)
	im = np.true_divide(im, 255)
	if i%images_per_class<use_images_per_class:
		data_train.append(np.concatenate((np.reshape(im[:28][:,:28], (784,)), [0])))
		label = []
		for k in range(0, number_of_classes):
			if(i//images_per_class != k):
				label.append(0)
			else:
				label.append(1)
		labels_train.append(label)
	elif i%number_of_classes>use_images_per_class:
		label = []
		for k in range(0, number_of_classes):
			if(i//images_per_class != k):
				label.append(0)
			else:
				label.append(1)
		data_test.append(np.concatenate((np.reshape(im[:28][:,:28], (784,)), [0])))
		labels_test.append(label)
	if i%images_per_class<use_augumented_images_per_class:
		data_train.append(np.concatenate((np.reshape(im[28:56][:,:28], (784,)), [1])))
		label = []
		for k in range(0, number_of_classes):
			if(i//images_per_class != k):
				label.append(0)
			else:
				label.append(1)
		labels_train.append(label)
	i+=1

def next_batch(num, data, labels):
    '''
    Return a total of `num` random samples and labels. 
    '''
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)	
	
learning_rate = 1e-4
batch_size = 32
n_iterations = 10000
dropout = 1
			
n_input = 785   # input layer (28x28 pixels)
n_hidden1 = 128 # 1st hidden layer
n_hidden2 = 64 # 2nd hidden layer
n_hidden3 = 128 # 3rd hidden layer
n_output = 400  # output layer

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32) 

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'w2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2], stddev=0.1)),
    'w3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden3, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'b2': tf.Variable(tf.constant(0.1, shape=[n_hidden2])),
    'b3': tf.Variable(tf.constant(0.1, shape=[n_hidden3])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
layer_3 = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
layer_drop = tf.nn.dropout(layer_3, keep_prob)
output_layer = tf.matmul(layer_3, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

pred_vec = output_layer

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = (tf.global_variables_initializer(), tf.local_variables_initializer())
sess = tf.Session()
sess.run(init)


for i in range(n_iterations):
	batch_x, batch_y = next_batch(batch_size, data_train, labels_train)
	sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # print loss and accuracy (per minibatch)
	if i%100==0:
		minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: data_train, Y: labels_train, keep_prob:1.0})
		test_accuracy = sess.run(accuracy, feed_dict={X: data_test, Y: labels_test, keep_prob:1.0})
		print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy), "\t| Accuracy on test set:", test_accuracy)

test_accuracy = sess.run(accuracy, feed_dict={X: data_test, Y: labels_test, keep_prob:1.0})
print("\nAccuracy on test set:", test_accuracy)
