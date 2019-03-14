import tensorflow as tf
import numpy as np
import pickle
import keras
import time
from scipy.io import wavfile
from python_speech_features import mfcc
from keras.models import *
from keras.layers import *

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

labels_input_global = None
input_length_global = None


def next_batch_gen(batch_size=32):
    while True:
        random_index = np.random.randint(0, num_examples, size=(batch_size,))
        batch_inputs = []  # 包含打了补丁的音频特征
        batch_mfccs = []  # 包含未打补丁的音频特征
        batch_labels = []  # 该list中每一项为一个内容如下(例子）的list：['one','space','five','space','three']
        max_label_len = 0
        batch_class_labels = []  # 每一项为['1', '10', '5', '10', '3', '-1', '-1'],
        # 其中-1是补丁，10是space的类，每一项的长度为max_label_len
        batch_labels_len = []  # 包含每个音频label的长度
        original_batch_labels = []  # 该list中每一项为'one five three'
        max_timestep_len = 0
        batch_timestep_len = []  # 包含每个音频的时间片长度

        # 提取mfccs
        for i in range(batch_size):
            samplerate, data = wavfile.read(wav_dir + '/' + str(random_index[i]) + '.wav')
            mfccs = mfcc(data, samplerate=samplerate, numcep=num_features)
            max_timestep_len = max(max_timestep_len, mfccs.shape[0])
            batch_mfccs.append(mfccs)

        # 给音频特征打补丁, 顺便打预处理label
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
            batch_labels_len.append(len(label))
            max_label_len = max(max_label_len, len(label))
            batch_labels.append(label)
        for k in range(batch_size):
            label = batch_labels[k]
            class_label = [dic[l] for l in label]
            for h in range(max_label_len - len(class_label)):
                class_label.append(-1)
            batch_class_labels.append(class_label)

        global labels_input_global, input_length_global
        labels_input_global = np.array(batch_class_labels)
        input_length_global = np.array(batch_timestep_len)
        yield ({
                   'inputs': np.array(batch_inputs),
                   'labels_input': np.array(batch_class_labels),
                   'input_length': np.array(batch_timestep_len),
                   'label_length': np.array(batch_labels_len),
               },
               np.ones(batch_size))


def print_decoded(decoded, labels_input):
    decoded_list = [reversed_dic[i] for i in np.asarray(decoded[1])]
    predict_label = [''] * decoded[2][0]
    decoded_index = np.asarray(decoded[0])
    for i in range(len(decoded_list)):
        if decoded_list[i] == 'space':
            predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + ' '
        else:
            predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + decoded_list[i]
    for j in range(decoded[2][0]):
        Original_label = ''
        for k in range(labels_input.shape[1]):
            digit_label = labels_input[j][k]
            if digit_label != -1:
                if digit_label == 10:
                    Original_label += ' '
                else:
                    Original_label += reversed_dic[digit_label]
        print('Original: ' + Original_label)
        print('Decoded:  ' + predict_label[j])


def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def def_model():
    inputs = Input(shape=(None, num_features), name='inputs')
    labels = Input(shape=(None,), name='labels_input')
    input_length = Input(shape=(1,), name='input_length')
    label_length = Input(shape=(1,), name='label_length')
    if bn_flag:
        inputs_norm = BatchNormalization(axis=-1)(inputs)
    else:
        inputs_norm = inputs
    x = inputs_norm
    for _ in range(num_layers):
        x = Bidirectional(LSTM(num_hidden, return_sequences=True), merge_mode='concat')(x)
    x = TimeDistributed(Dense(50, activation='tanh'))(x)
    x = TimeDistributed(Dense(num_classes, activation='softmax', name='Dense2'))(x)
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])
    # base_model = Model(input=inputs, output=x)

    model = Model(input=[inputs, labels, input_length, label_length], output=[ctc_loss])
    return model


model = def_model()
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='rmsprop')
output_layers = ['Dense2']
metrics_name = 'hack'
model.metrics_names += [metrics_name]
model.metrics_tensors += [layer.output for layer in model.layers if layer.name in output_layers]


class my_callback(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs=None):
        Dense2_output = K.function([model.layers[0].input], [model.layers[3].output])
        decoded = K.ctc_decode(Dense2_output, input_length_global, greedy=False, beam_width=10)
        print_decoded(decoded, labels_input_global)


my_callback_object = my_callback()

model.fit_generator(next_batch_gen(batch_size), steps_per_epoch=num_batches_per_epoch,
                    epochs=num_epochs, callbacks=[my_callback_object], shuffle=False)
