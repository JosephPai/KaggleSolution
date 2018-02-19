# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# print(test.head())

import tensorflow as tf
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop('label',axis=1),train['label'],stratify=train['label'])
X_train_tf = []
for j in range(len(X_train)):
    X_train_tf.append(np.array([X_train.values[j][28*i:i+28] for i in range(28)]))

print(len(X_train_tf),len(X_train_tf[0]),len(X_train_tf[0][0]))
X_train_tf = np.array(X_train_tf)
print(X_train_tf.shape)

X_test_tf = []
for j in range(len(X_test)):
    X_test_tf.append(np.array([X_test.values[j][28*i:i+28] for i in range(28)]))

print(len(X_test_tf),len(X_test_tf[0]),len(X_test_tf[0][0]))
X_test_tf = np.array(X_test_tf)

train_dataset = X_train.values
train_labels = y_train
valid_dataset = X_test.values
valid_labels = y_test
test_dataset = test.values
test_labels = y_train[:len(test.values)]

image_size = 28
num_lables = 10
num_channels = 1

def reformat(dataset, labels):
    dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
    labels = (np.arange(num_lables) == labels[:,None]).astype(np.float32)
    return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Vliadation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])

def prediction(predictions):
    return np.argmax(predictions, 1)

batch_size = 16
patch_size = 5
depth = 16
num_hidden = 64

graph = tf.Graph()

with graph.as_default():
    # input data
    tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_lables))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_valid_labels = tf.constant(valid_labels)
    tf_test_dataset = tf.constant(test_dataset)

    # variable
    layer1_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, num_channels, depth], stddev=0.1))
    layer1_biases = tf.Variable(tf.zeros([depth]))
    layer2_weights = tf.Variable(tf.truncated_normal([patch_size, patch_size, depth, depth], stddev=0.1))
    layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))
    layer3_weights = tf.Variable(tf.truncated_normal([image_size//4*image_size//4*depth, num_hidden], stddev=0.1))
    layer3_biases = tf.Variable(tf.constant(1.0, shape=[num_hidden]))
    layer4_weights = tf.Variable(tf.truncated_normal([num_hidden, num_lables], stddev=0.1))
    layer4_biases = tf.Variable(tf.constant(1.0, shape=[num_lables]))

    # model
    def model(data):
        conv = tf.nn.conv2d(data, layer1_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer1_biases)
        conv = tf.nn.conv2d(hidden, layer2_weights, [1, 2, 2, 1], padding='SAME')
        hidden = tf.nn.relu(conv + layer2_biases)
        shape = hidden.get_shape().as_list()
        reshape = tf.reshape(hidden, [shape[0], shape[1]*shape[2]*shape[3]])
        hidden = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
        return tf.matmul(hidden, layer4_weights) + layer4_biases

    # training computation
    logits = model(tf_train_dataset)
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf_train_labels, logits=logits))\
        +0.001*(tf.nn.l2_loss(layer1_weights) + tf.nn.l2_loss(layer1_biases) + \
            tf.nn.l2_loss(layer2_weights) + tf.nn.l2_loss(layer2_biases) + \
            tf.nn.l2_loss(layer3_weights) + tf.nn.l2_loss(layer3_biases))

    # optimizer
    optimizer = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

    # predictions for the training, validation and test data
    train_prediction = tf.nn.softmax(logits)
    valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
    test_prediction = tf.nn.softmax(model(tf_test_dataset))

    num_steps = 7001
    hard_steps = 101

    with tf.Session(graph=graph) as session:
        tf.global_variables_initializer().run()
        print('Initialized')
        for step in range(num_steps//2):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset: (offset+batch_size), :, :, :]
            batch_labels = train_labels[offset: (offset+batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels:batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.lf%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.lf%%' % accuracy(valid_prediction.eval(), valid_labels))
        val_guess = prediction(valid_prediction.eval())

        is_hard = lambda i: val_guess[i] != np.argmax(valid_labels[i])
        hard_indices = []
        for i in range(len(valid_labels)):
            if is_hard(i):
                hard_indices.append(i)
        hard_dataset = valid_dataset[hard_indices]
        hard_labels = valid_labels[hard_indices]

        for step in range(hard_steps):
            offset = (step * batch_size) % (hard_labels.shape[0] - batch_size)
            batch_data = hard_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = hard_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
            _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step %50 ==0):
                print('Minibatch loss at step %d: %f of extra-hard training' % (step, 1))
                print('Minibatch accuracy: %.lf%%'% accuracy(predictions, batch_labels))
                print('Validation accuracy: %.lf%%' % accuracy(valid_prediction.eval(), valid_labels))
        for step in range(num_steps//2):
            offset = (step * batch_size + num_steps) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 500 == 0):
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                print('Validation accuracy: %.1f%%' % accuracy(
                    valid_prediction.eval(), valid_labels))
        val_guess = prediction(valid_prediction.eval())
        print('Test accuracy(invalid): %.lf%%' % accuracy(test_prediction.eval(), test_labels))
        Y_pred = prediction(test_prediction.eval())

    submission = pd.DataFrame({
        "ImageId": range(1, len(Y_pred) + 1),
        "Label": Y_pred
    })
    submission.to_csv('MNist.csv', index=False)

