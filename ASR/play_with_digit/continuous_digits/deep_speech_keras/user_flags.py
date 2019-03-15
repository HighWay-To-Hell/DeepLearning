from absl import flags


def def_flags():
    flags.DEFINE_bool(
        name="resume_train_from_ckpt",
        default=True,
        help="如果model_ckpt_dir中有ckpt, 则从检查点继续训练"
    )
    flags.DEFINE_bool(
        name="predict",
        default=True,
        help="只以predict模式运行模型，请保证model_ckpt_dir中有保存好的weights, 同时注意修改ler_test_batch_num这个flag"
    )
    """跟目录相关的"""
    flags.DEFINE_string(
        name="model_ckpt_dir",
        default=r"model_ckpt/",
        help="保存模型检查点的目录，请在结尾带上斜杆"
    )
    flags.DEFINE_string(
        name="saved_model_dir",
        default=r"saved_model/",
        help="保存训练完的模型的目录, 请在结尾带上斜杠"
    )
    flags.DEFINE_string(
        name="train_csv_path",
        default=r"D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20\train/syn_csv.csv",
        help="训练数据的csv文件的路径"
    )
    flags.DEFINE_string(
        name="test_csv_path",
        default=r"D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20\test/syn_csv.csv",
        help="测试数据的csv文件的路径"
    )
    flags.DEFINE_string(
        name="vocabulary_file",
        default=r"D:\DAYDAYUP\ASR\data\corpus\syn_continuous_digit_soft_noise_3_20/vocabulary.txt",
        help="指向“词汇表”，请在其中包含空格，参考该py文件同名目录下的vocabulary.txt"
    )

    """跟模型相关的"""
    flags.DEFINE_integer(
        name="rnn_hidden_size",
        default=200,
        help="The hidden size of RNNs."
    )
    flags.DEFINE_integer(
        name="rnn_layers",
        default=2,
        help="RNN的层数"
    )
    flags.DEFINE_bool(
        name="rnn_is_bidirectional",
        default=True,
        help="是否使用双向rnn"
    )
    flags.DEFINE_bool(
        name="use_bias_in_last_dense",
        default=True,
        help="是否在最后一层全连接中使用bias"
    )

    """跟训练有关的"""
    flags.DEFINE_float(
        name="learning_rate",
        default=5e-4,
        help="lr,已加了learning_rate decay, 若loss没有下降，会自动降低lr(new_lr = 0.7 * lr)，直到最小值0.0001"
    )
    flags.DEFINE_integer(
        name="epochs",
        default=20,
        help="max epochs"
    )
    flags.DEFINE_integer(
        name="batch_size",
        default=32,
        help="batch_size"
    )
    flags.DEFINE_integer(
        name="ler_test_batch_num",
        default=48,
        help="在训练过程，每50个step，会用测试集测一下ler，测试集中用到的总数为ler_test_batch_num * batch_size,"
             "若乘积大于测试集的数量，会从头取。"
    )
    flags.DEFINE_float(
        name="ler_threshold",
        default=0.05,
        help="当测试集的ler达到要求，便会停止训练"
    )
    flags.DEFINE_bool(
        name="sortagrad",
        default=True,
        help="如果为真，会将训练集按时长排序，建议默认为真，有利于收敛，见http://www.tbluche.com/ctc_and_blank.html"

    )
    return flags.FLAGS
