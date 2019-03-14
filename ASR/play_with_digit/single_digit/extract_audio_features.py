import numpy as np
import pickle
import os
import scipy.io.wavfile as wav
from python_speech_features import mfcc

'''
抽取音频的13个倒谱特征并利用pickle写入硬盘，
文件夹目录结构如下所示，其中每个英文数字的目录下，包含对应的wav格式单个数字读音的音频。
      ─wav
        ├─eight
        ├─five
        ├─four
        ├─nine
        ├─one
        ├─seven
        ├─six
        ├─three
        ├─two
        └─zero
'''

wav_dir = r'../wav'  # 指向wav文件夹
dump_dir = r'data'  # 指向pickle的输出目录
num_train = 3
num_dev = 3
num_test = 3


# num_train + num_dev: num_train + num_dev + num_test 应小于总文件数

def get_mfcc_label(wav_dir):
    """

    :param
    wav_dir -- 对应上面目录树中wav的路径
    :returns
    mfccs -- python列表，len(mfccs) = m为总音频个数，列表中每一项为一个音频的全部特征
    labels -- python列表，包含与mfccs中的音频特征的label，位置相互对应
    """

    mfccs = []
    labels = []
    list1 = os.listdir(wav_dir)  # 获取wav_dir下的子目录list1
    for i in range(0, len(list1)):
        path1 = os.path.join(wav_dir, list1[i])  # 拼接路径为"wav_dir\list1[i]
        list2 = os.listdir(path1)  # 获取wav_dir\list1[i]下的文件名list2
        for j in range(0, len(list2)):
            path2 = os.path.join(path1, list2[j])  # 拼接路径为"wav_dir\list1[i]\list2[i],即为某个音频的路径
            samplerate, data = wav.read(path2)

            # 音频中绝大部分为1秒整的音频，小部分不足一秒或超过一秒，由于本次实验主要是为了测试FNN的效果，故直接抛弃这些不是1秒整的音频
            if samplerate == 16000 and len(data) == 16000:
                feature_mfcc = mfcc(data, samplerate, numcep=13)
                feature_mfcc_flatten = feature_mfcc.flatten()
                mfccs.append(feature_mfcc_flatten)
                labels.append(list1[i])  # wav_dir\list1[i]\list2[i]这个音频对应的label为list1[i]
    return mfccs, labels


def get_X_Y(mfccs, labels):
    """
    将mfccs, labels转为np.ndarray, 并把labels中的label转为one hot vector

    :param mfccs: python列表，len(mfccs) = m为总音频个数，列表中每一项为一个音频的全部特征
    :param labels: python列表，包含与mfccs中的音频特征的label，位置相互对应
    :return X: np.ndarray, shape = (m, num of features)
    :return Y: np.ndarray, shape = (m, num of classes)
    """

    X = np.array(mfccs)
    Y = np.zeros((len(labels), 10))

    dict = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9}

    for i in range(0, len(labels)):
        Y[i, dict[labels[i]]] = 1

    return X, Y


def shuffle_split_save(X, Y):
    """
    先打乱X,Y, 再将X,Y分为训练集，开发集，测试集， 并写入硬盘
    :param X: np.ndarray, shape = (m, num of features)
    :param Y: np.ndarray, shape = (m, num of classes)
    :return:
    """
    state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)

    X_train = X[0:num_train]
    X_dev = X[num_train: num_train + num_dev]
    X_test = X[num_train + num_dev: num_train + num_dev + num_test]

    Y_train = Y[0:num_train]
    Y_dev = Y[num_train: num_train + num_dev]
    Y_test = Y[num_train + num_dev: num_train + num_dev + num_test]

    if not os.path.exists(dump_dir):
        os.mkdir(dump_dir)

    with open(dump_dir + '/X_train', 'wb') as f:
        pickle.dump(X_train, f)
    with open(dump_dir + '/Y_train', 'wb') as f:
        pickle.dump(Y_train, f)
    with open(dump_dir + '/X_dev', 'wb') as f:
        pickle.dump(X_dev, f)
    with open(dump_dir + '/Y_dev', 'wb') as f:
        pickle.dump(Y_dev, f)
    with open(dump_dir + '/X_test', 'wb') as f:
        pickle.dump(X_test, f)
    with open(dump_dir + '/Y_test', 'wb') as f:
        pickle.dump(Y_test, f)

    print('done!')


mfccs, labels = get_mfcc_label(wav_dir)
X, Y = get_X_Y(mfccs, labels)
shuffle_split_save(X, Y)
