from keras import backend as K
from keras.layers import Input, Conv2D, ReLU, BatchNormalization, Reshape, LSTM, \
    Bidirectional, TimeDistributed, Dense, Lambda
from keras import Model

CONV_FILTERS = 32
BN_EPSILON = 1e-5
BN_DECAY = 0.997


class MyModel(object):
    def __init__(self, flags_obj, num_classes):
        self.rnn_hidden_size = flags_obj.rnn_hidden_size
        self.rnn_layers = flags_obj.rnn_layers
        self.rnn_is_bidirectional = flags_obj.rnn_is_bidirectional
        self.use_bias_in_last_dense = flags_obj.use_bias_in_last_dense
        self.num_classes = num_classes
        self.model_for_predict, self.model_for_train = self.build_model()

    def get_model_for_predict(self):
        return self.model_for_predict

    def get_model_for_train(self):
        return self.model_for_train

    def build_model(self):
        features = Input(shape=(None, 129), dtype="float32", name="features")
        seq_len_after_conv = Input(shape=(1,), dtype="int32", name="seq_len_after_conv")
        labels = Input(shape=(None,), dtype="int32", name="labels")
        labels_len = Input(shape=(1,), dtype="int32", name="labels_len")

        # readme （1）
        seq_len_after_conv_equal = Lambda(self.equal_layer)(seq_len_after_conv)
        labels_equal = Lambda(self.equal_layer)(labels)
        labels_len_equal = Lambda(self.equal_layer)(labels_len)
        # 将features扩张一个维度,以便送进cnn
        # readme （2）
        x = Lambda(self.expand_dim)(features)

        # cnn
        # 请注意，如果要修改kernel_size, stride, 要同时修改dataset中的compute_length_after_conv函数
        x = self.cnn_layer(x, kernel_size=(41, 11), stride=(2, 2), layer_id=1)
        x = self.cnn_layer(x, kernel_size=(21, 11), stride=(2, 1), layer_id=2)

        # 经过cnn后，x的shape为（batch_size, timesteps, features, channels),
        # 需要reshape成(batch_size, timesteps, features*channels)以便送进rnn中
        # readme (3)
        x = Reshape((-1, K.int_shape(x)[2] * CONV_FILTERS), name="reshape_after_cnn2")(x)
        # RNN
        for i in range(self.rnn_layers):
            x = self.rnn_layer(x, i)

        # readme(4)
        logits = TimeDistributed(Dense(self.num_classes, activation="softmax",
                                       use_bias=self.use_bias_in_last_dense))(x)
        model_for_predict = Model(inputs=[features, seq_len_after_conv, labels, labels_len],
                                  outputs=[logits, seq_len_after_conv_equal, labels_equal, labels_len_equal])

        ctc_loss = Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [labels, logits, seq_len_after_conv, labels_len])
        model_for_train = Model(inputs=[features, seq_len_after_conv, labels, labels_len], outputs=[ctc_loss])

        return model_for_predict, model_for_train

    def cnn_layer(self, layer_input, kernel_size, stride, layer_id):
        x = Conv2D(kernel_size=kernel_size, strides=stride, filters=CONV_FILTERS,
                   padding="same", use_bias=False, name="cnn_" + str(layer_id))(layer_input)
        x = ReLU(max_value=6, name="relu_after_cnn_" + str(layer_id))(x)
        x = self.bn_layer(x, name="bn_after_cnn_" + str(layer_id))
        return x

    def rnn_layer(self, layer_input, layer_id):
        if self.rnn_is_bidirectional:
            x = Bidirectional(LSTM(units=self.rnn_hidden_size, return_sequences=True), name="blstm_" + str(layer_id))(
                layer_input)
            x = self.bn_layer(x, name="bn_after_blstm_" + str(layer_id))
        else:
            x = LSTM(self.rnn_hidden_size, return_sequences=True, name="lstm_" + str(layer_id))(layer_input)
            x = self.bn_layer(x, name="bn_after_lstm_" + str(layer_id))
        return x

    def bn_layer(self, input, name):
        x = BatchNormalization(momentum=BN_DECAY, epsilon=BN_EPSILON, name=name)(input)
        return x

    def ctc_lambda_func(self, args):
        labels, logits, seq_len_after_conv, labels_len = args
        return K.ctc_batch_cost(y_true=labels, y_pred=logits,
                                input_length=seq_len_after_conv, label_length=labels_len)

    def expand_dim(self, args):
        return K.expand_dims(args, axis=-1)

    def equal_layer(self, args):
        return args
