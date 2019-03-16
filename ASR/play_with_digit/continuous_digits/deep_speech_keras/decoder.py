import tensorflow as tf
import numpy as np
import keras.backend as K


def decoder(logits, seq_len_after_conv, labels, labels_len, text_f):
    logits = tf.cast(logits, dtype="float32")
    logits = tf.transpose(logits, [1, 0, 2])  # time major

    # readme (4)
    logits = tf.log(logits + tf.keras.backend.epsilon())
    seq_len_after_conv = tf.squeeze(seq_len_after_conv, axis=-1)
    seq_len_after_conv = tf.cast(seq_len_after_conv, dtype="int32")

    # 解码出来decoded[j]才是一个sparse tensor， 代表第j条最可能路径
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len_after_conv, beam_width=2,
                                               merge_repeated=False)

    labels_len_squeeze = tf.squeeze(labels_len, axis=1)
    # labels_len_squeeze = tf.constant(labels_len)
    labels_sparse = K.ctc_label_dense_to_sparse(labels, labels_len_squeeze)
    labels_sparse = tf.cast(labels_sparse, dtype="int64")
    ler = tf.reduce_mean(tf.edit_distance(decoded[0], labels_sparse))
    with tf.Session() as sess:
        decoded, ler = sess.run([decoded[0], ler])

    # 现在decoded是一个sparse array, decoded[1]是values, values.shape=(total_label_len, 1)
    # total_label_len = sum over len of tokens of each sample
    decoded_list = [text_f.index_to_token[i] for i in np.asarray(decoded[1])]
    predict_label = [''] * decoded[2][0]  # decoded[2]是一个shape，=(num of samples, max_label_len)

    # decoded[0]是indices, indices.shape=(total_label_len, 2), 第一列指示sample，第二列指示token
    decoded_indices = np.asarray(decoded[0])
    for i in range(len(decoded_list)):
        predict_label[decoded_indices[i, 0]] = predict_label[decoded_indices[i, 0]] + decoded_list[i] + ' '

    original_label = [""] * labels.shape[0]
    for i in range(len(original_label)):
        for j in range(labels_len[i][0]):
            original_label[i] = original_label[i] + text_f.index_to_token[labels[i][j]] + ' '
    for i in range(len(original_label)):
        print("Original: " + original_label[i])
        print("Decoded:  " + predict_label[i])
    return ler
