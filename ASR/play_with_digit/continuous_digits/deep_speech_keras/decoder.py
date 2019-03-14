import tensorflow as tf
import numpy as np
import keras.backend as K


def decoder(logits, seq_len_after_conv, labels, labels_len, text_f):
    logits = np.asarray(logits, dtype=float)
    logits = np.transpose(logits, [1, 0, 2])
    logits = tf.log(logits + tf.keras.backend.epsilon())
    logits = tf.cast(logits, dtype="float32")
    seq_len_after_conv = tf.squeeze(seq_len_after_conv, axis=-1)
    seq_len_after_conv = tf.cast(seq_len_after_conv, dtype="int32")
    decoded, _ = tf.nn.ctc_beam_search_decoder(logits, seq_len_after_conv, beam_width=2,
                                               merge_repeated=False)
    labels_len_squeeze = tf.squeeze(labels_len)
    labels_sparse = K.ctc_label_dense_to_sparse(labels, labels_len_squeeze)
    labels_sparse = tf.cast(labels_sparse, dtype="int64")
    ler = tf.reduce_mean(tf.edit_distance(decoded[0], labels_sparse))
    with tf.Session() as sess:
        decoded, ler = sess.run([decoded[0], ler])
    decoded_list = [text_f.index_to_token[i] for i in np.asarray(decoded[1])]
    predict_label = [''] * decoded[2][0]
    decoded_index = np.asarray(decoded[0])
    for i in range(len(decoded_list)):
        predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + decoded_list[i] + ' '
    original_label = [""] * labels.shape[0]
    for i in range(len(original_label)):
        for j in range(labels_len[i][0]):
            original_label[i] = original_label[i] + text_f.index_to_token[labels[i][j]] + ' '
    for i in range(len(original_label)):
        print("Original: " + original_label[i])
        print("Decoded:  " + predict_label[i])
    return ler
