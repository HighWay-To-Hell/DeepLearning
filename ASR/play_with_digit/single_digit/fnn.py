import math
import numpy as np
import datetime
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops

data_dir = 'data'


def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_x -- scalar, size of an feature vector
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [n_x, None] and dtype "float"
    Y -- placeholder for the input labels, of shape [n_y, None] and dtype "float"

    Tips:
    - You will use None because it let's us be flexible on the number of examples you will for the placeholders.
      In fact, the number of examples during test/train is different.
    """

    X = tf.placeholder(tf.float32, shape=[n_x, None])
    Y = tf.placeholder(tf.float32, shape=[n_y, None])

    return X, Y


def initialize_parameters():
    """
    initialize weights
    :return: parameters -- a python dict containing weights
    """
    W1 = tf.get_variable("W1", [99, 1287], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [99, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [10, 99], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [10, 1], initializer=tf.zeros_initializer())
    #     W3 = tf.get_variable("W3", [13,99], initializer = tf.contrib.layers.xavier_initializer())
    #     b3 = tf.get_variable("b3", [13,1], initializer = tf.zeros_initializer())
    #     W4 = tf.get_variable("W4", [10,13], initializer = tf.contrib.layers.xavier_initializer())
    #     b4 = tf.get_variable("b4", [10,1], initializer = tf.zeros_initializer())

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    #                   "W3": W3,
    #                   "b3": b3,
    #                   "W4": W4,
    #                   "b4": b4
    return parameters


def forward_propagation(X, parameters, keep_prob):
    """
    Implements the forward propagation for the model

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters
    keep_prob -- input keep_prob placeholder

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    #     W3 = parameters['W3']
    #     b3 = parameters['b3']
    #     W4 = parameters['W4']
    #     b4 = parameters['b4']

    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    A1 = tf.nn.dropout(A1, keep_prob)
    Z2 = tf.add(tf.matmul(W2, A1), b2)
    #     A2 = tf.nn.relu(Z2)                                               # A2 = relu(Z2)
    #     A2 = tf.nn.dropout(A2, keep_prob)
    #     Z3 = tf.add(tf.matmul(W3, A2), b3)                                              # Z3 = np.dot(W3,Z2) + b3
    #     A3 = tf.nn.relu(Z3)
    #     A3 = tf.nn.dropout(A3, keep_prob)
    #     Z4 = tf.add(tf.matmul(W4, A3), b4)

    return Z2


def compute_cost(Z2, Y, reg_term):
    """
    Computes the cost

    Arguments:
    Z3 -- output of forward propagation (output of the last LINEAR unit)
    Y -- "true" labels vector placeholder, same shape as Z3
    reg_term -- L2_regularizer term

    Returns:
    cost - Tensor of the cost function
    """

    # to fit the tensorflow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
    logits = tf.transpose(Z2)
    labels = tf.transpose(Y)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)) + reg_term

    return cost


def random_mini_batches(X, Y, mini_batch_size=64):
    """
    Creates a list of random minibatches from (X, Y)

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" ,  of shape (number of classes, number of examples)
    mini_batch_size - size of the mini-batches, integer

    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """

    m = X.shape[1]  # number of training examples
    mini_batches = []

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches


def model(X_train, Y_train, X_dev, Y_dev, learning_rate=0.0001, keep_prob_value=0.9, reg_scale=0.01,
          num_epochs=500, minibatch_size=32):
    """
    Implements a tensorflow neural network

    Arguments:
    X_train -- training set, of shape (features size, number of training examples)
    Y_train -- test set, of shape (number of classes, number of training examples)
    X_dev -- test set, of shape (features size, number of test examples)
    Y_dev -- test set, of shape (number of classes, number of test examples)
    learning_rate -- learning rate of the optimization
    keep_prob_value -- value of keep_prob
    reg_scale -- L2 regularization scale
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    # To keep track of the training/test cost/accuracy
    train_costs = []
    test_costs = []
    train_accuracys = []
    test_accuracys = []

    # Create Placeholders
    X, Y = create_placeholders(n_x, n_y)
    keep_prob = tf.placeholder(tf.float32)

    # Initialize parameters
    parameters = initialize_parameters()

    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z2 = forward_propagation(X, parameters, keep_prob)

    # Add all weights into tf.GraphKeys.WEIGHTS
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W1'])
    tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W2'])
    #     tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W3'])
    #     tf.add_to_collection(tf.GraphKeys.WEIGHTS, parameters['W4'])

    # Create regularizer
    regularizer = tf.contrib.layers.l2_regularizer(scale=reg_scale)
    # Create regularization term
    reg_term = tf.contrib.layers.apply_regularization(regularizer)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z2, Y, reg_term)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Calculate the correct predictions
    correct_prediction = tf.equal(tf.argmax(Z2), tf.argmax(Y))

    # Calculate accuracy on the test set
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:

        writer = tf.summary.FileWriter('.', sess.graph)
        # Run the initialization
        sess.run(init)

        # Print some info
        print('features size', n_x, '  num of training set: ', m, '  num of test set ', X_dev.shape[1])
        print('num_epochs = %f , learning_rate = %f , lambd_value = %f , keep_prob = %f '
              % (num_epochs, learning_rate, reg_scale, keep_prob_value))

        # Do the training loop
        for epoch in range(num_epochs):

            minibatches = random_mini_batches(X_train, Y_train, minibatch_size)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y) = minibatch

                # IMPORTANT: The line that runs the graph on a minibatch.
                # Run the session to execute the "optimizer", the feedict should contain a minibatch for (X,Y).
                sess.run(optimizer, feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob: keep_prob_value})

            # Print the cost/accuracy every 10 epoch
            if epoch % 10 == 0 or epoch == (num_epochs - 1):
                train_cost = sess.run(cost, feed_dict={X: X_train, Y: Y_train, keep_prob: 1})
                test_cost = sess.run(cost, feed_dict={X: X_dev, Y: Y_dev, keep_prob: 1})
                train_accuracy = accuracy.eval({X: X_train, Y: Y_train, keep_prob: 1})
                test_accuracy = accuracy.eval({X: X_dev, Y: Y_dev, keep_prob: 1})
                train_accuracys.append(train_accuracy)
                test_accuracys.append(test_accuracy)
                test_costs.append(test_cost)
                train_costs.append(train_cost)
                print("(cost, dev_cost, train_accuracy, test_accuracy) after epoch %i: (%f, %f, %f, %f)" % (
                    epoch, train_cost, test_cost, train_accuracy, test_accuracy))

        # plot the cost
        plt.plot(np.squeeze(train_costs))
        plt.plot(np.squeeze(test_costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per 10)')
        plt.title(
            "Learning rate =" + str(learning_rate) + " keep_prob = " + str(keep_prob_value) + " reg_scale = " + str(
                reg_scale))
        plt.show()

        # plot the accuracy
        plt.plot(np.squeeze(train_accuracys))
        plt.plot(np.squeeze(test_accuracys))
        plt.ylabel('accuracy')
        plt.xlabel('iterations (per 10)')
        plt.title(
            "Learning rate =" + str(learning_rate) + " keep_prob = " + str(keep_prob_value) + " reg_scale = " + str(
                reg_scale))
        plt.show()

        # save the parameters in a variable
        parameters = sess.run(parameters)

        # save parameters to local
        with open('parameters', 'wb') as f:
            pickle.dump(parameters, f)

        print("Parameters saved!")

        writer.close()

        return parameters


# Load dataset
with open(data_dir + '/X_train', 'rb') as f:
    X_train = pickle.load(f).T
with open(data_dir + '/Y_train', 'rb') as f:
    Y_train = pickle.load(f).T
with open(data_dir + '/X_dev', 'rb') as f:
    X_dev = pickle.load(f).T
with open(data_dir + '/Y_dev', 'rb') as f:
    Y_dev = pickle.load(f).T

# Train
starttime = datetime.datetime.now()
parameters = model(X_train, Y_train, X_dev, Y_dev,
                   num_epochs=100, learning_rate=0.00005,
                   reg_scale=0.026, keep_prob_value=0.7, minibatch_size=32)
endtime = datetime.datetime.now()
print((endtime - starttime))
