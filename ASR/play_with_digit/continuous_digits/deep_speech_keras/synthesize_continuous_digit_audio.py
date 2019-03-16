from scipy.io import wavfile
from random import randint
import numpy as np
import os
import pickle

wav_dir = r'/content/drive/My Drive/Colab/wav/single'  # wav根目录
output_dir = r'/content/drive/My Drive/Colab/wav/continuous/train'  # 输出目录
# num_of_syn_audio = 256  # 要合成的连续数字语音个数
# min_syn_sec = 3  # 合成语音最短时长(s), 这里为了处理方便，令其大于max_digit_num
# max_syn_sec = 3  #
# min_digit_num = 2  # 每个合成语音里最少包含几个数字
# max_digit_num = 2  #

if not os.path.exists(output_dir):
    os.mkdir(output_dir)
if not os.path.exists(output_dir + '/syn_wav'):
    os.mkdir(output_dir + '/syn_wav')


def get_random_time_segment(segment_len, background_len):
    """
    Gets a random time segment of duration segment_len in (0, background_len).

    Arguments:
    segment_len -- the duration of the audio clip

    Returns:
    segment_time -- a tuple of (segment_start, segment_end)
    """

    segment_start = np.random.randint(low=0,
                                      high=background_len - segment_len)  # Make sure segment doesn't run past the background
    segment_end = segment_start + segment_len - 1

    return (segment_start, segment_end)


def is_overlapping(segment_time, previous_segments):
    """
    Checks if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment_time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise
    """

    segment_start, segment_end = segment_time

    # Step 1: Initialize overlap as a "False" flag.
    overlap = False

    # Step 2: loop over the previous_segments start and end times.
    # Compare start/end times and set the flag to True if there is an overlap
    for previous_start, previous_end in previous_segments:
        if segment_start <= previous_end and segment_end >= previous_start:
            overlap = True

    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    Insert a new audio segment over the background noise at a random time step, ensuring that the
    audio segment does not overlap with existing segments.

    Arguments:
    background -- a background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- times where audio segments have already been placed

    Returns:
    syn_audio -- the updated background audio
    """

    segment_len = len(audio_clip)
    background_len = len(background)

    # Step 1: Use one of the helper functions to pick a random time segment onto which to insert
    # the new audio clip.
    segment_time = get_random_time_segment(segment_len, background_len)

    # Step 2: Check if the new segment_time overlaps with one of the previous_segments. If so, keep
    # picking new segment_time at random until it doesn't overlap.
    while is_overlapping(segment_time, previous_segments):
        segment_time = get_random_time_segment(segment_len, background_len)

    # Step 3: Add the new segment_time to the list of previous_segments
    previous_segments.append(segment_time)

    # Step 4: Superpose audio segment and background
    syn_audio = background
    syn_audio[segment_time[0]: segment_time[0] + segment_len] += audio_clip
    return syn_audio


def get_digit_wav_data_labels(wav_dir):
    wav_data = []
    labels = []
    list1 = os.listdir(wav_dir)  # 获取wavdir下的子目录list1
    for i in range(0, len(list1)):
        path1 = os.path.join(wav_dir, list1[i])  # 拼接路径为"wavdir\list1[i]
        list2 = os.listdir(path1)  # 获取wavdir\list1[i]下的文件名list2
        for j in range(0, len(list2)):
            path2 = os.path.join(path1, list2[j])  # 拼接路径为"wavdir\list1[i]\list2[i],即为某个音频的路径
            _, data = wavfile.read(path2)
            wav_data.append(data)
            labels.append(list1[i])  # wavdir\list1[i]\list2[i]这个音频对应的label为list1[i]
    return wav_data, labels


def write_label_to_disk(syn_labels, output_dir):
    file = open(output_dir + '/syn_labels.txt', 'a')
    for i in range(len(syn_labels)):
        file.write(syn_labels[i] + '\n')
    file.close()
    # with open(output_dir + '/syn_labels_pickle', 'wb') as f:
    #     pickle.dump(syn_labels, f)


def synthesize_continuous_digit_audio(wav_dir, output_dir):
    wav_data, labels = get_digit_wav_data_labels(wav_dir)
    wav_data_len = len(wav_data)
    # syn_labels = []  # 对应的label，例如'one three four'

    for curr_num in range(27, 31):
        syn_labels = []
        min_syn_sec = curr_num  # 合成语音最短时长(s), 这里为了处理方便，令其大于max_digit_num
        max_syn_sec = curr_num  #
        if curr_num <= 10:
            num_of_syn_audio = 1024 - 128 * (curr_num - 4)  # 要合成的连续数字语音个数
            min_digit_num = curr_num - 4  # 每个合成语音里最少包含几个数字
            max_digit_num = curr_num - 4  #
        else:
            num_of_syn_audio = 256  # 要合成的连续数字语音个数
            min_digit_num = curr_num - 11  # 每个合成语音里最少包含几个数字
            max_digit_num = curr_num - 11  #
        for i in range(num_of_syn_audio):
            digit_num = randint(min_digit_num, max_digit_num)  # 随机数字的个数, 这里randint包含最大值
            syn_sec = randint(min_syn_sec, max_syn_sec)  # 随机音频的时长
            background = np.random.randint(-20, 20, size=(syn_sec * 16000,), dtype='int16')  # int16对应16-bit PCM
            previous_segments = []  # 记录下数字被插入的位置
            wav_index = np.random.randint(0, wav_data_len, size=(digit_num,))  # 随机数字音频的索引，这里randint不包含最大值
            while max(len(wav_data[index]) for index in wav_index) > 16000:
                wav_index = np.random.randint(0, wav_data_len, size=(digit_num,))
            # 开始向background插入数字
            syn_audio = background
            for j in range(digit_num):
                syn_audio = insert_audio_clip(syn_audio, wav_data[wav_index[j]], previous_segments)

            # 根据previous_segments打label
            # np.argsort返回结果例子：[2, 0, 3, 1]
            # 其中1代表previous_segments[0, 0]在previous_segments[:, 0]中按从小到大排序是第2位
            # 也就是说wav_index[3]所对应的数字label，在syn_label中排在第2
            previous_segments_argsort = np.argsort(np.array(previous_segments)[:, 0])
            label_list = [0] * digit_num
            for k in range(digit_num):
                label_list[np.where(previous_segments_argsort == k)[0][0]] = labels[wav_index[k]]
            syn_label = ' '.join(label_list)

            wavfile.write(output_dir + '/syn_wav/' + str(curr_num) + '-' + str(i + 1) + '.wav', 16000, syn_audio)
            syn_labels.append(str(curr_num) + '-' + str(i + 1) + ' ' + syn_label)

        write_label_to_disk(syn_labels, output_dir)


synthesize_continuous_digit_audio(wav_dir, output_dir)
