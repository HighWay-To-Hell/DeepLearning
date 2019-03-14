from keras.utils import Sequence
import math
import random
from scipy import signal
from scipy.io import wavfile
import numpy as np
import ASR.play_with_digit.continuous_digits.deep_speech_keras.textfeaturizer as textfeaturizer


class DataGenerator(Sequence):

    def __init__(self, vocabulary, sortagrad, csv_path, mode, batch_size):
        self.text_featurizer = textfeaturizer.TextFeaturizer(vocabulary)
        self.sortagrad = sortagrad
        self.csv_path = csv_path
        self.mode = mode
        self.batch_size = batch_size
        self.entries = self.get_entries()

    def __len__(self):
        return int(math.ceil(len(self.entries) / self.batch_size))

    def __getitem__(self, idx):
        features, seq_len = self.get_spectrogram_and_seq_len(idx)
        seq_len_after_conv = self.compute_length_after_conv(seq_len)
        labels, labels_len = self.get_labels_and_labels_len(idx)
        return ({
                    "features": features,
                    "seq_len_after_conv": seq_len_after_conv,
                    "labels": labels,
                    "labels_len": labels_len
                },
                np.zeros((features.shape[0], 1)))

    def on_epoch_end(self):
        # 在一次epoch之后，每次epoch结束都batch_wise的打乱训练集
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
        features = []
        seq_len = []
        if (idx + 1) * self.batch_size > len(self.entries):
            idx_range = range(idx * self.batch_size, len(self.entries))
        else:
            idx_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        max_timesteps = 0
        for i in idx_range:
            wav_path = self.entries[i][0]
            sample_rate, samples = wavfile.read(wav_path)
            _, _, spectrogram = signal.spectrogram(samples, sample_rate)
            spectrogram = np.transpose(spectrogram)
            features.append(spectrogram)
            seq_len.append(spectrogram.shape[0])
            if spectrogram.shape[0] > max_timesteps:
                max_timesteps = spectrogram.shape[0]
            features = self.pad_features(features, max_timesteps)
        return np.asarray(features, dtype="float32"), np.asarray(seq_len, dtype="int32")

    def get_labels_and_labels_len(self, idx):
        labels = []
        labels_len = []
        if (idx + 1) * self.batch_size > len(self.entries):
            idx_range = range(idx * self.batch_size, len(self.entries))
        else:
            idx_range = range(idx * self.batch_size, (idx + 1) * self.batch_size)
        max_labels_len = 0
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
        with open(self.csv_path, "r") as f:
            entries = f.read().splitlines()[1:]
        entries = [line.split("\t", 2) for line in entries]
        if self.sortagrad and self.mode == "train":
            entries.sort(key=lambda item: int(item[1]))
        return entries

    def pad_features(self, features, max_timesteps):
        frequencies = features[0].shape[1]
        for i in range(len(features)):
            curr_timesteps = features[i].shape[0]
            if curr_timesteps < max_timesteps:
                patch = np.zeros((max_timesteps - curr_timesteps, frequencies))
                features[i] = np.vstack(features[i], patch)
        return features

    def pad_labels(self, labels, max_labels_len):
        for i in range(len(labels)):
            curr_label_len = len(labels[i])
            for j in range(max_labels_len - curr_label_len):
                labels[i].append(-1)
        return labels

    def compute_length_after_conv(self, seq_len):
        seq_len = ((seq_len - 41) / 2).astype(dtype="int16") + 1
        seq_len = ((seq_len - 21) / 2).astype(dtype="int16") + 1
        return seq_len
