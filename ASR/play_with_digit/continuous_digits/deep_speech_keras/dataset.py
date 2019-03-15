from keras.utils import Sequence
import math
import random
from scipy import signal
from scipy.io import wavfile
import numpy as np
import ASR.play_with_digit.continuous_digits.deep_speech_keras.textfeaturizer as textfeaturizer


class DataGenerator(Sequence):

    def __init__(self, vocabulary, sortagrad, csv_path, mode, batch_size):
        """
        keras model.fit_generator 所要求的data_gen
        :param vocabulary:
        :param sortagrad:
        :param csv_path:
        :param mode: 'train' or 'test'
        :param batch_size:
        """
        self.text_featurizer = textfeaturizer.TextFeaturizer(vocabulary)
        self.sortagrad = sortagrad
        self.csv_path = csv_path
        self.mode = mode
        self.batch_size = batch_size
        # TODO 详细解释一下这里
        self.count = 0  # 由于keras model.predict的缺陷。需要自己维护一个idx
        self.entries = self.get_entries()

    def __len__(self):
        """
        要求实现的两个函数之一
        :return: 返回一个epoch中有多少个batch
        """
        return int(math.ceil(len(self.entries) / self.batch_size))

    def __getitem__(self, idx):
        """
        要求实现的两个函数之一，返回一个batch
        :param idx: 取第几批数据
        :return: (X, Y)
                X:
                features with shape=(batch_size, max_time_steps, frequencies)
                seq_len_after_conv with shape=(batch_size, 1) contain 'time steps len' after conv of each sample
                labels with shape=(batch_size, max_labels_len)
                labels_len with shape=(batch_size, 1) contain true label len of each sample
                Y:
                useless in ctc model, just pass some fake data

        """

        features, seq_len = self.get_spectrogram_and_seq_len(self.count)
        # 之所以不用原来的seq_len， 见model.py的build_model
        seq_len_after_conv = self.compute_length_after_conv(seq_len)
        labels, labels_len = self.get_labels_and_labels_len(self.count)
        if self.count == int(math.ceil(len(self.entries) / self.batch_size)):
            self.count = 0
        return ({
                    "features": features,
                    "seq_len_after_conv": seq_len_after_conv,
                    "labels": labels,
                    "labels_len": labels_len
                },
                np.zeros((features.shape[0], 1)))

    def on_epoch_end(self):
        """
        覆盖父类方法
        每次epoch结束都batch_wise的打乱训练集, 这样也避免了在第一次epoch打乱训练集，因为第一次根据时长排序进行训练有利收敛
        :return:
        """

        if self.mode == "train":
            shuffled_entries = []
            max_buckets = int(math.floor(len(self.entries) / self.batch_size))
            buckets = [i for i in range(max_buckets)]
            random.shuffle(buckets)
            for i in buckets:
                shuffled_entries.extend(self.entries[i * self.batch_size: (i + 1) * self.batch_size])
            shuffled_entries.extend(self.entries[max_buckets * self.batch_size:])
            self.entries = shuffled_entries

    def get_spectrogram_and_seq_len(self, idx):
        """
        如果修改signal.spectrogram的参数，要同时修改model中features的shape
        :param idx:
        :return: features with shape=(batch_size, max_time_steps, frequencies)
                seq_len with shape=(batch_size, 1) contain true time steps len of each sample
        """
        features = []
        seq_len = []
        if (idx + 1) * self.batch_size > len(self.entries):
            idx_range = range(idx * self.batch_size, len(self.entries))
        else:
            idx_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        max_time_steps = 0  # for padding
        for i in idx_range:
            wav_path = self.entries[i][0]
            sample_rate, samples = wavfile.read(wav_path)
            _, _, spectrogram = signal.spectrogram(samples, sample_rate)
            # spectrogram original shape=(frequencies, time_steps)
            spectrogram = np.transpose(spectrogram)
            features.append(spectrogram)
            seq_len.append(spectrogram.shape[0])
            if spectrogram.shape[0] > max_time_steps:
                max_time_steps = spectrogram.shape[0]
            features = self.pad_features(features, max_time_steps)
        return np.asarray(features, dtype="float32"), np.asarray(seq_len, dtype="int32")

    def get_labels_and_labels_len(self, idx):
        """

        :param idx:
        :return: labels with shape=(batch_size, max_labels_len)
                labels_len with shape=(batch_size, 1) contain true label len of each sample
        """
        labels = []
        labels_len = []
        if (idx + 1) * self.batch_size > len(self.entries):
            idx_range = range(idx * self.batch_size, len(self.entries))
        else:
            idx_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        max_labels_len = 0  # for padding
        for i in idx_range:
            text = self.entries[i][2]
            text = text.strip().lower().split(" ")
            labels.append([self.text_featurizer.token_to_index[token] for token in text])
            labels_len.append(len(text))
            if max_labels_len < len(text):
                max_labels_len = len(text)
            labels = self.pad_labels(labels, max_labels_len)
        return np.asarray(labels, dtype="int32"), np.asarray(labels_len, dtype="int32")

    def get_entries(self):
        """
        get entries from csv file
        :return:
        """
        with open(self.csv_path, "r") as f:
            entries = f.read().splitlines()[1:]
        entries = [line.split("\t", 2) for line in entries]
        if self.sortagrad and self.mode == "train":
            entries.sort(key=lambda item: int(item[1]))
        return entries

    def pad_features(self, features, max_time_steps):
        frequencies = features[0].shape[1]
        for i in range(len(features)):
            curr_time_steps = features[i].shape[0]
            if curr_time_steps < max_time_steps:
                patch = np.zeros((max_time_steps - curr_time_steps, frequencies))
                features[i] = np.vstack((features[i], patch))
        return features

    def pad_labels(self, labels, max_labels_len):
        for i in range(len(labels)):
            curr_label_len = len(labels[i])
            for j in range(max_labels_len - curr_label_len):
                labels[i].append(-1)
        return labels

    def compute_length_after_conv(self, seq_len):
        seq_len_after_conv_1 = ((seq_len - 41) / 2).astype(dtype="int16") + 1
        seq_len_after_conv = ((seq_len_after_conv_1 - 21) / 2).astype(dtype="int16") + 1
        return seq_len_after_conv
