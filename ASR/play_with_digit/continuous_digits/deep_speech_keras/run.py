import dataset, textfeaturizer, model, user_flags, decoder
import keras
from keras.utils import plot_model
from absl import app as absl_app
import os
from keras import optimizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


def train():
    if not os.path.exists(flags_obj.model_ckpt_dir):
        os.mkdir(flags_obj.model_ckpt_dir)
    if not os.path.exists(flags_obj.saved_model_dir):
        os.mkdir(flags_obj.saved_model_dir)
    text_f = textfeaturizer.TextFeaturizer(flags_obj.vocabulary_file)
    my_model = model.MyModel(flags_obj, num_classes=len(text_f.tokens))
    model_for_predict = my_model.get_model_for_predict()
    model_for_train = my_model.get_model_for_train()
    train_data_gen = dataset.DataGenerator(vocabulary=flags_obj.vocabulary_file,
                                           sortagrad=flags_obj.sortagrad,
                                           csv_path=flags_obj.train_csv_path,
                                           mode="train",
                                           batch_size=flags_obj.batch_size)
    test_data_gen = dataset.DataGenerator(vocabulary=flags_obj.vocabulary_file,
                                          sortagrad=False,
                                          csv_path=flags_obj.test_csv_path,
                                          mode="test",
                                          batch_size=flags_obj.batch_size)
    optimizer = optimizers.RMSprop(lr=flags_obj.learning_rate)
    model_for_train.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    plot_model(model_for_predict, to_file='model_for_predict.png', show_shapes=True)
    plot_model(model_for_train, to_file='model_for_train.png', show_shapes=True)

    early_stop_on_ler = EarlyStopingBaseOnLer(ler_threshold=flags_obj.ler_threshold, test_data_gen=test_data_gen,
                                              model_for_predict=model_for_predict, text_f=text_f,
                                              ler_test_batch_num=flags_obj.ler_test_batch_num)

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.7, patience=3, min_lr=0.0001, mode='min',
                                  cooldown=2, verbose=1)
    checkpointer = ModelCheckpoint(monitor='loss', mode='min', verbose=1, save_best_only=True, save_weights_only=True,
                                   filepath=flags_obj.model_ckpt_dir + "weights.h5"
                                   )
    # tensorboard = keras.callbacks.TensorBoard(log_dir='tb_log_dir')
    if flags_obj.resume_train_from_ckpt:
        if os.path.exists(flags_obj.model_ckpt_dir + "weights.h5"):
            model_for_train.load_weights(flags_obj.model_ckpt_dir + "weights.h5")
    model_for_train.fit_generator(train_data_gen, epochs=flags_obj.epochs, verbose=1,
                                  callbacks=[early_stop_on_ler, reduce_lr, checkpointer])
    model_for_train.save_weights(flags_obj.model_ckpt_dir + "weights_after_train_stopped.h5")
    model_for_predict.save(flags_obj.saved_model_dir + "model_for_predict.h5")
    model_for_train.save(flags_obj.saved_model_dir + "model_for_train.h5")


class EarlyStopingBaseOnLer(EarlyStopping):
    def __init__(self, model_for_predict, test_data_gen, text_f, ler_test_batch_num=1, monitor="val_loss",
                 patience=1000,
                 verbose=1,
                 mode="auto",
                 ler_threshold=0.05,
                 ):
        super(EarlyStopingBaseOnLer, self).__init__()
        self.ler_threshold = ler_threshold
        self.count = 0
        self.model_for_predict = model_for_predict
        self.test_data_gen = test_data_gen
        self.text_f = text_f
        self.ler_test_batch_num = ler_test_batch_num

    def on_batch_end(self, batch, logs=None):
        self.count += 1
        if self.count == 50:
            print("\ntesting ler...")
            ler = 0
            for i in range(flags_obj.ler_test_num):
                print("testing " + str(i) + 'th sample')
                (logits, seq_len_after_conv, labels, labels_len) = \
                    self.model_for_predict.predict_generator(self.test_data_gen, steps=1, verbose=1)
                curr_ler = decoder.decoder(logits, seq_len_after_conv, labels, labels_len, text_f=self.text_f)
                print("curr_ler: " + str(curr_ler) + "\n")
                ler += curr_ler
            ler /= flags_obj.ler_test_num
            print("ler: " + str(ler))
            if ler <= self.ler_threshold:
                self.model.stop_training = True
            self.count = 0

    def on_epoch_end(self, epoch, logs=None):
        return


def predict():
    text_f = textfeaturizer.TextFeaturizer(flags_obj.vocabulary_file)
    my_model = model.MyModel(flags_obj, num_classes=len(text_f.tokens))
    model_for_predict = my_model.get_model_for_predict()
    test_data_gen = dataset.DataGenerator(vocabulary=flags_obj.vocabulary_file,
                                          sortagrad=False,
                                          csv_path=flags_obj.test_csv_path,
                                          mode="test",
                                          batch_size=flags_obj.batch_size)
    if os.path.exists(flags_obj.model_ckpt_dir + "weights_after_train_stopped.h5"):
        model_for_predict.load_weights(flags_obj.model_ckpt_dir + "weights_after_train_stopped.h5")
    elif os.path.exists(flags_obj.model_ckpt_dir + "weights.h5"):
        model_for_predict.load_weights(flags_obj.model_ckpt_dir + "weights.h5")
    else:
        print(str(flags_obj.model_ckpt_dir) + "目录下无保存好的weights")
        return
    print("\npredicting...")
    ler = 0
    num_of_entries = test_data_gen.__len__()
    for i in range(num_of_entries):
        print("predicting " + str(i) + 'th sample')
        (logits, seq_len_after_conv, labels, labels_len) = \
            model_for_predict.predict_generator(test_data_gen, steps=1, verbose=0)
        curr_ler = decoder.decoder(logits, seq_len_after_conv, labels, labels_len, text_f=text_f)
        print("curr_ler: " + str(curr_ler) + "\n")
        ler += curr_ler
    ler = ler / num_of_entries
    print("Ler: " + str(ler))


def main(_):
    if flags_obj.predict:
        predict()
    else:
        train()


if __name__ == "__main__":
    flags_obj = user_flags.def_flags()
    absl_app.run(main)
