import tensorflow as tf
import numpy as np
import pickle
import time
from scipy.io import wavfile
from python_speech_features import mfcc
from keras import backend as K
from keras.layers import StackedRNNCells, RNN, LSTMCell, concatenate, Dense, BatchNormalization

num_features = 13
num_classes = 10 + 1 + 1  # 10 + space + blank label，10 + space是真实label，blank label是tf api要求加的

unit_test_flag = True
bn_flag = True

num_epochs = 500
num_hidden = 100
num_layers = 1

# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
initial_learning_rate = 0.1
decay_steps = 40
decay_rate = 0.9

lstm_kernel_dropout = 0
lstm_recurrent_dropout = 0

# batch_size = 32 时，每批batch耗时9秒，如果5000个音频全用上，1个epoch耗时23.43分钟，10个epoch耗时3.9个小时，而且数据集越大，收敛越慢
# 只用一个音频测试时，即batch_size, num_examples = 1，收敛需要200左右epoch
batch_size = 32
num_examples = 32  #
num_batches_per_epoch = int(num_examples / batch_size)

wav_dir = r'D:\学在华科\自然语言处理\语音识别\资料\音频\合成连续数字-白噪音\syn_audios'
pickle_label_path = r'D:\学在华科\自然语言处理\语音识别\资料\音频\合成连续数字-白噪音\syn_labels_pickle'

dic = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
       'seven': 7, 'eight': 8, 'nine': 9, 'space': 10}

reversed_dic = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: 'space'}

with open(pickle_label_path, 'rb') as f:
    str_label = pickle.load(f)


def next_batch(wav_dir, str_label, model):
    assert (model == 'train' or model == 'test')
    random_index = np.random.randint(0, num_examples, size=(batch_size,))
    batch_inputs = []  # 用于存放打好补丁的音频特征数据
    batch_mfccs = []  # 用于存放没打补丁的音频特征数据
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

        # 三种打补丁的方式：
        # 1，补零，适用于没有使用batch_normalization, 和测试的情况
        # 2，重复前面(从开始位置)的帧，适用于使用batch_normalization训练的情况
        #       注意这种方式，要避免每批中音频长度差别太大，
        # 3，补上环境音的mfcc！！！！
        # https://github.com/SeanNaren/deepspeech.pytorch/issues/312
        if model == 'test' or (bn_flag == False and model == 'train'):
            patch = np.zeros((max_timestep_len - current_timestep_len, num_features))
        else:
            patch = mfccs[:(max_timestep_len - current_timestep_len)]
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
        print('Decoded:  ' + predict_label[j] + '\n')


sess = tf.Session()
K.set_session(sess)

# e.g: log filter bank or MFCC features
# Has size [batch_size, max_step_size, num_features], but the
# batch_size and max_step_size can vary along each step
inputs = tf.placeholder(tf.float32, [None, None, num_features])

BN = BatchNormalization()
inputs_norm = BN(inputs)

# Here we use sparse_placeholder that will generate a
# SparseTensor required by ctc_loss op.
targets = tf.sparse_placeholder(tf.int32)

# 1d array of size [batch_size]
seq_len = tf.placeholder(tf.int32, [None])

# Defining the cell
lstm_cells = []
lstm_cells_backward = []

for _ in range(num_layers):
    lstm_cells.append(LSTMCell(num_hidden, dropout=lstm_kernel_dropout,
                               recurrent_dropout=lstm_recurrent_dropout))
    lstm_cells_backward.append(LSTMCell(num_hidden, dropout=lstm_kernel_dropout,
                                        recurrent_dropout=lstm_recurrent_dropout))

# Stacking rnn cells
stack_lstm_cell = StackedRNNCells(lstm_cells)
stack_lstm_cell_backward = StackedRNNCells(lstm_cells_backward)

# The second output is the last state and we will no use that
LSTM_layer = RNN(stack_lstm_cell, return_sequences=True)
LSTM_layer_backward = RNN(stack_lstm_cell_backward, return_sequences=True, go_backwards=True)

lstm_output = LSTM_layer(inputs_norm)
lstm_output_backward = LSTM_layer_backward(inputs_norm)
lstm_output_concatenate = concatenate([lstm_output, lstm_output_backward], axis=-1)

blstm_output = tf.reshape(lstm_output_concatenate, [-1, num_hidden * 2])

dense_layer_1 = Dense(50, activation='relu')
dense_layer_2 = Dense(num_classes)

# Doing the affine projection
dense_layer_1_output = dense_layer_1(blstm_output)
logits = dense_layer_2(dense_layer_1_output)

# Reshaping back to the original shape
logits = tf.reshape(logits, [tf.shape(inputs)[0], -1, num_classes])

# Time major
logits = tf.transpose(logits, (1, 0, 2))

loss = tf.nn.ctc_loss(targets, logits, seq_len)
cost = tf.reduce_mean(loss)

# Option 2: tf.contrib.ctc.ctc_beam_search_decoder
# (it's slower but you'll get better results)
decoded, log_prob = tf.nn.ctc_beam_search_decoder(logits, seq_len)

# Inaccuracy: label error rate
ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
                                      targets))

# optimizer = tf.train.AdamOptimizer().minimize(cost)
# optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                           global_step, decay_steps=decay_steps, decay_rate=decay_rate,
                                           staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(cost, global_step)

with sess.as_default():
    sess.run(tf.global_variables_initializer())

    for curr_epoch in range(num_epochs):
        train_cost = train_ler = 0
        epoch_start_time = time.time()

        for batch in range(num_batches_per_epoch):
            batch_start_time = time.time()
            train_inputs, train_targets, \
            train_seq_len, original_batch_labels = next_batch(wav_dir, str_label, 'train')

            feed = {inputs: train_inputs,
                    targets: train_targets,
                    seq_len: train_seq_len,
                    K.learning_phase(): 1}

            batch_cost, d, batch_ler, _ = sess.run([cost, decoded[0], ler, optimizer], feed)
            train_cost += batch_cost * batch_size
            train_ler += batch_ler * batch_size
            batch_end_time = time.time()
            print_decoded(d, original_batch_labels)
            log = "Epoch {}/{}, batch {}/{}, batch_cost = {:.3f}, batch_ler = {:.3f}"
            print(log.format(curr_epoch + 1, num_epochs, batch, num_batches_per_epoch,
                             batch_cost, batch_ler) + ', batch_time_cost = ' + str(
                batch_end_time - batch_start_time) + '\n')

        if unit_test_flag == False:
            train_cost /= num_examples
            train_ler /= num_examples

            val_inputs, val_targets, val_seq_len, \
            original_batch_labels = next_batch(wav_dir, str_label, 'test')
            val_feed = {inputs: val_inputs,
                        targets: val_targets,
                        seq_len: val_seq_len,
                        K.learning_phase(): 0}

            val_cost, val_ler, d = sess.run([cost, ler, decoded[0]], feed_dict=val_feed)

            print_decoded(d, original_batch_labels)

            log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
                  "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"
            print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                             val_cost, val_ler, time.time() - epoch_start_time))
