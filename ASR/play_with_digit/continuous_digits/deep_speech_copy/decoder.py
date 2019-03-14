import tensorflow as tf
import numpy as np


def decoder(pred, entries):
    input_seq_len = np.zeros((len(pred),))
    input_seq_len += pred[0].shape[0]
    # pred = tf.transpose(pred, [1, 0, 2])
    pred = np.asarray(pred)
    pred = np.transpose(pred, [1, 0, 2])
    targets = [entry[2] for entry in entries]
    pred = tf.log(pred + tf.keras.backend.epsilon())
    decoded, _ = tf.nn.ctc_beam_search_decoder(pred, input_seq_len, beam_width=3, merge_repeated=False)
    index_to_token = {0: 'zero', 1: 'one', 2: 'two', 3: 'three', 4: 'four', 5: 'five',
                      6: 'six', 7: 'seven', 8: 'eight', 9: 'nine', 10: '_'}
    with tf.Session() as sess:
        decoded = sess.run(decoded[0])
    decoded_list = [index_to_token[i] for i in np.asarray(decoded[1])]
    predict_label = [''] * decoded[2][0]
    decoded_index = np.asarray(decoded[0])
    for i in range(len(decoded_list)):
        predict_label[decoded_index[i, 0]] = predict_label[decoded_index[i, 0]] + decoded_list[i] + '_'
    for j in range(decoded[2][0]):
        tf.logging.info('Original: ' + targets[j])
        tf.logging.info('Decoded:  ' + predict_label[j] + '\n')
