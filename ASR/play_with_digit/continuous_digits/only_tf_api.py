import tensorflow as tf
import numpy as np
import pickle
import time
from scipy.io import wavfile
from python_speech_features import mfcc

'''

'''
num_features = 13
num_classes = 10 + 1 + 1  # 10 + space + blank label，10 + space是真实label，blank label是tf api要求加的

num_epochs = 100
num_hidden = 100
num_layers = 1
learning_rate = 0.1
lstm_kernel_dropout = 0
lstm_recurrent_dropout = 0
batch_size = 1

bn_flag = True

num_examples = 1
num_batches_per_epoch = int(num_examples / batch_size)

wav_dir = r'D:\学在华科\自然语言处理\语音识别\资料\音频\合成连续数字-白噪音\syn_audios'
pickle_label_path = r'D:\学在华科\自然语言处理\语音识别\资料\音频\合成连续数字-白噪音\syn_labels_pickle'

dic = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
       'seven': 7, 'eight': 8, 'nine': 9, 'space': 10}

reversed_dic = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'space'}

with open(pickle_label_path, 'rb') as f:
    str_label = pickle.load(f)


def next_train_batch(wav_dir, str_label):
    random_index = np.random.randint(0, num_examples, size=(batch_size,))
    batch_inputs = []
    batch_mfccs = []
    batch_labels = []  # 该list中每一项为一个内容如下(例子）的list：['one','space','five','space','three']
    original_batch_labels = []  # 该list中每一项为'one five three'
    max_timestep_len = 0
    batch_timestep_len = []
    for i in range(batch_size):
        samplerate, data = wavfile.read(wav_dir + '/' + str(random_index[i]) + '.wav')
        mfccs = mfcc(data, samplerate=samplerate, numcep=num_features)
        if max_timestep_len < mfccs.shape[0]: max_timestep_len = mfccs.shape[0]
        batch_mfccs.append(mfccs)
    for j in range(batch_size):
        mfccs = batch_mfccs[j]
        current_timestep_len = mfccs.shape[0]
        batch_timestep_len.append(current_timestep_len)
        patch = np.zeros((max_timestep_len - current_timestep_len, num_features))
        batch_inputs.append(np.row_stack((mfccs, patch)))

        # 处理label
        label = str_label[random_index[j]].replace(' ', ' space ')
        original_batch_labels.append(str_label[random_index[j]])
        label = label.split(' ')
        batch_labels.append(label)
    return np.array(batch_inputs), batch_labels_list_to_sparse(batch_labels), batch_timestep_len, original_batch_labels


def batch_labels_list_to_sparse(batch_labels):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
       See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/ctc/ctc_loss_op_test.py
       for example of SparseTensor format'''

    indices = []
    vals = []
    for tI, target in enumerate(batch_labels):
        for seqI, val in enumerate(target):
            indices.append([tI, seqI])
            vals.append(dic[val])
    shape = [len(batch_labels), np.asarray(indices).max(0)[1] + 1]
    return (np.array(indices), np.array(vals), np.array(shape))


def print_decoded(decoded, original_batch_labels):
    decoded_list = [reversed_dic[i] for i in np.asarray(decoded[1])]
    predict_label = [''] * decoded[2][0]
    decoded_index = np.asarray(decoded[0])
    for i in range(len(decoded_list)):
        if decoded_list[i] == 'space':
            predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + ' '
        else:
            predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + decoded_list[i]
    for j in range(decoded[2][0]):
        print('Original: ' + original_batch_labels[j])
        print('Decoded: ' + predict_label[j])


graph = tf.Graph()
with graph.as_default():
    # e.g: log filter bank or MFCC features
    # Has size [batch_size, max_step_size, num_features], but the
    # batch_size and max_step_size can vary along each step
    inputs = tf.placeholder(tf.float32, [None, None, num_features])

    bn_flag_pl = tf.placeholder(tf.bool, shape=())
    inputs_norm = tf.layers.batch_normalization(inputs, training=bn_flag)

    # Here we use sparse_placeholder that will generate a
    # SparseTensor required by ctc_loss op.
    targets = tf.sparse_placeholder(tf.int32)

    # 1d array of size [batch_size]
    seq_len = tf.placeholder(tf.int32, [None])

    # Defining the cell
    # Can be:
    #   tf.nn.rnn_cell.RNNCell
    #   tf.nn.rnn_cell.GRUCell
    lstm_cells = []
    lstm_cells_backward = []

    for _ in range(num_layers):
        lstm_cells.append(
            tf.keras.layers.LSTMCell(num_hidden, dropout=lstm_kernel_dropout, recurrent_dropout=lstm_recurrent_dropout))
        lstm_cells_backward.append(
            tf.keras.layers.LSTMCell(num_hidden, dropout=lstm_kernel_dropout, recurrent_dropout=lstm_recurrent_dropout))

    # Stacking rnn cells
    stack_lstm_cell = tf.keras.layers.StackedRNNCells(lstm_cells)
    stack_lstm_cell_backward = tf.keras.layers.StackedRNNCells(lstm_cells_backward)

    # The second output is the last state and we will no use that
    LSTM_layer = tf.keras.layers.RNN(stack_lstm_cell, return_sequences=True)
    LSTM_layer_backward = tf.keras.layers.RNN(stack_lstm_cell_backward,
                                              return_sequences=True, go_backwards=True)

    lstm_output = LSTM_layer(inputs)
    lstm_output_backward = LSTM_layer_backward(inputs)
    lstm_output_concatenate = tf.keras.layers.concatenate([lstm_output, lstm_output_backward], axis=-1)

    blstm_output = tf.reshape(lstm_output_concatenate, [-1, num_hidden * 2])

    dense_layer = tf.keras.layers.Dense(num_classes)
    # shape = tf.shape(inputs)
    # batch_s, max_time_steps = shape[0], shape[1]
    #
    # # Reshaping to apply the same weights over the timesteps
    # outputs = tf.reshape(outputs, [-1, num_hidden * 2])

    # Truncated normal with mean 0 and stdev=0.1
    # Tip: Try another initialization
    # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
    # W = tf.Variable(tf.truncated_normal([num_hidden * 2,
    #                                      num_classes],
    #                                     stddev=0.1))
    # Zero initialization
    # Tip: Is tf.zeros_initializer the same?
    # b = tf.Variable(tf.constant(0., shape=[num_classes]))

    # Doing the affine projection
    logits = dense_layer(blstm_output)

    # Reshaping back to the original shape
    logits = tf.reshape(logits, [tf.shape(inputs)[0], -1, num_classes])

    # Time major
    logits = tf.transpose(logits, (1, 0, 2))

    loss = tf.nn.ctc_loss(targets, logits, seq_len)
    cost = tf.reduce_mean(loss)

    # optimizer = tf.train.AdamOptimizer().minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
    # (it's slower but you'll get better results)
    decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

    # Inaccuracy: label error rate
    ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                          targets))

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        epoch_start_time = time.time()

        for batch in range(num_batches_per_epoch):
            batch_start_time = time.time()
            train_inputs, train_targets, train_seq_len, \
            original_batch_labels = next_train_batch(wav_dir, str_label)

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len,
                    bn_flag_pl: bn_flag}

            batch_cost, d, batch_ler, _ = session.run([cost, decoded[0], ler, optimizer], feed)
            train_cost += batch_cost * batch_size
            train_ler += batch_ler * batch_size

            print_decoded(d, original_batch_labels)
            batch_end_time = time.time()
            log = "Epoch {}/{}, batch {}/{}, batch_cost = {:.3f}, batch_ler = {:.3f}"
            print(log.format(curr_epoch + 1, num_epochs, batch, num_batches_per_epoch,
                             batch_cost, batch_ler) + ', used_time = ' + str(batch_end_time - batch_start_time) + '\n')
        # train_cost /= num_examples
        # train_ler /= num_examples
        #
        # val_inputs, val_targets, val_seq_len, original_batch_labels = next_train_batch(wav_dir, str_label)
        # val_feed = {inputs: val_inputs,
        #             targets: val_targets,
        #             seq_len: val_seq_len,
        #             bn_flag_pl: bn_flag}
        #
        # val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)
        #
        # # Decoding
        # d = session.run(decoded[0], feed_dict=val_feed)
        # print_decoded(d, original_batch_labels)
        #
        # log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
        #       "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
        # print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
        #                  val_cost, val_ler, time.time() - start))
