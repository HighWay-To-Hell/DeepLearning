import numpy as np


def compute_spectrogram_feature(samples, sample_rate, stride_ms=10.0,
                                window_ms=20.0, max_freq=None, eps=1e-14):
    """Compute the spectrograms for the input samples(waveforms).

    More about spectrogram computation, please refer to:
    https://en.wikipedia.org/wiki/Short-time_Fourier_transform.
    """
    if max_freq is None:
        max_freq = sample_rate / 2
    if max_freq > sample_rate / 2:
        raise ValueError("max_freq must not be greater than half of sample rate.")

    if stride_ms > window_ms:
        raise ValueError("Stride size must not be greater than window size.")

    stride_size = int(0.001 * sample_rate * stride_ms)
    window_size = int(0.001 * sample_rate * window_ms)

    # Extract strided windows
    truncate_size = (len(samples) - window_size) % stride_size
    samples = samples[:len(samples) - truncate_size]
    nshape = (window_size, (len(samples) - window_size) // stride_size + 1)
    nstrides = (samples.strides[0], samples.strides[0] * stride_size)
    windows = np.lib.stride_tricks.as_strided(
        samples, shape=nshape, strides=nstrides)
    assert np.all(
        windows[:, 1] == samples[stride_size:(stride_size + window_size)])

    # Window weighting, squared Fast Fourier Transform (fft), scaling
    weighting = np.hanning(window_size)[:, None]
    fft = np.fft.rfft(windows * weighting, axis=0)
    fft = np.absolute(fft)
    fft = fft ** 2
    scale = np.sum(weighting ** 2) * sample_rate
    fft[1:-1, :] *= (2.0 / scale)
    fft[(0, -1), :] /= scale
    # Prepare fft frequency list
    freqs = float(sample_rate) / window_size * np.arange(fft.shape[0])

    # Compute spectrogram feature
    ind = np.where(freqs <= max_freq)[0][-1] + 1
    specgram = np.log(fft[:ind, :] + eps)
    return np.transpose(specgram, (1, 0))


class AudioFeaturizer(object):
    """Class to extract spectrogram features from the audio input."""

    def __init__(self,
                 sample_rate=16000,
                 window_ms=20.0,
                 stride_ms=10.0):
        self.sample_rate = sample_rate
        self.window_ms = window_ms
        self.stride_ms = stride_ms


def compute_label_feature(text, token_to_idx):
    """convert string to a list of integers"""
    # tokens = text.replace(' ', ' _ ')
    tokens = text.split('_')
    feats = [token_to_idx[token] for token in tokens]
    return feats


class TextFeaturizer(object):
    def __init__(self):
        self.token_to_index = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
                               'seven': 7, 'eight': 8, 'nine': 9}
        self.index_to_token = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                               6: 'six', 7: 'seven', 8: 'eight', 9: 'nine'}
        self.speech_labels = ['zero', 'one', 'two', 'three', 'four', 'five', 'six',
                              'seven', 'eight', 'nine', '_']
