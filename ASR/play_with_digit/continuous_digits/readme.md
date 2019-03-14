#3月7号：
1：
    数据的准备：
    音频特征：
        每个batch，从wav目录下，随机挑取batch_size个音频，抽取mfcc特征，由于每个音频的时长不同，所以导致每个
        音频的timesteps不同，引出了两个问题：
        一：如果timestep相同，网络的输入shape=(batch_size, timesteps, num_features), 那么timestep不同要如何
        把这一批的音频特征数据存放到一个3维矩阵呢？
        二：rnn的要如何构建？
        解决方法：
            老版的tf api的解决方法是，人工预先补零，也就是输入的shape=（b_s,max_timesteps, n_f)，
            并记录下每个音频的timestep，存放在seq_len，这是个size=batch_size的1D array。构建rnn时，先定义rnncell(或者LSTMCell等),
            然后利用outputs,_ = tf.nn.dynamic_rnn(rnncell, inputs, seq_len)，这个api的意思是，传入shape为（b_s,max_timesteps, n_f)
            的inputs, 和seq_len，这样，这个rnn网络在计算的时候就会根据seq_len中的值，只利用有效的时间片计算，那些补零的
            时间片就不会利用到，(不过好像根据翻到的博文，上面的解释是就放着为零就是了，具体忘了）
            而新版的api，tf.keras.layers.RNN, 老版的api已经提示换到这个api，并且说，两个功能完全一样，但是这个新版的api
            去掉了seq_len的输入，stackoverflow上有关于这个的提问https://stackoverflow.com/questions/54989442/rnn-in-tensorflow-vs-keras-depreciation-of-tf-nn-dynamic-rnn
            ，还没人回答，现在只好将补零的直接输入。
        新解决方法：
            三种打补丁的方式：
            1，补零，适用于没有使用batch_normalization, 和测试的情况
            2，重复前面(从开始位置)的帧，适用于使用batch_normalization训练的情况
                  注意这种方式，要避免每批中音频长度差别太大，
            3，补上环境音的mfcc！！！！
            https://github.com/SeanNaren/deepspeech.pytorch/issues/312
        tf.keras.layers.RNN 的使用：
            直接from keras.layers import RNN ，然后，RNN(rnncell) ,这样是错误的
            应该是import tensorflow as tf, 然后tf.keras.layers.RNN(rnncell)，这样才行，tensorflow更喜欢你导入整体，而不喜欢导入部分，
            因为导入部分，意味着要访问私有域
            详情见：https://github.com/tensorflow/tensorflow/issues/15736#issuecomment-354667237
        同时用keras跟tf见：https://blog.keras.io/keras-as-a-simplified-interface-to-tensorflow-tutorial.html
    
    音频label:
        tf.nn.ctc_loss所要求的label(target)输入是一个sparsetensor，也就是稀疏张量，内含元组（index, values, shape)
        生成方法如下：
            batch_labels=[]
            取每个音频的转写，将其split后，存放在一个list中，e.g: 'one five three' 变成label=['one', ' ', 'five', ' ', 'three']
            注意空格' '
            batch_labels.append(label)
            batch_labels应该长这样：
            [['one', ' ', 'five', ' ', 'three'],
             ['two', ' ', 'six']]
            然后利用这个函数(见下面)batch_labels_list_to_sparse(batch_labels)返回所需的元组，其中，shape的值为(batch_size, max_label_len)
            每个音频的转写长度不一样，并没有关系
        注意：在batch_labels_list_to_sparse(batch_labels)中，会把one，two，' ',等变成对应的class的值，在本例中，zero-nine是1-9
            空格' '是10，然后存进稀疏张量的values中
        注意：不用one-hot表示label！！
    
    关于keras的backend中的ctc_batch_cost：
        keras.backend.ctc_batch_cost(
            y_true,
            y_pred,
            input_length,
            label_length
            )
        y_true: tensor (samples, max_string_length) containing the truth labels.
        y_pred: tensor (samples, time_steps, num_categories) containing the prediction, or output of the softmax.
        input_length: tensor (samples, 1) containing the sequence length for each batch item in y_pred.
        label_length: tensor (samples, 1) containing the sequence length for each batch item in y_true.
        这里，label_length的解释是sequence length for each batch item in y_true，其实应该是label_length,看源码：
        
        from tensorflow.python.ops import ctc_ops as ctc
        ...
        def ctc_batch_cost(y_true, y_pred, input_length, label_length):
              """Runs CTC loss algorithm on each batch element.
              Returns:
                  Tensor with shape (samples,1) containing the
                      CTC loss of each element.
              """
              label_length = math_ops.to_int32(array_ops.squeeze(label_length))
              input_length = math_ops.to_int32(array_ops.squeeze(input_length))
              sparse_labels = math_ops.to_int32(
                  ctc_label_dense_to_sparse(y_true, label_length))
    
              y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
    
              return array_ops.expand_dims(
                  ctc.ctc_loss(
                      inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)
    
        这里面，先对数据进行处理，然后调用了tensorflow的api，关键的是这句ctc_label_dense_to_sparse(y_true, label_length), 看这个函数的源码
        def ctc_label_dense_to_sparse(labels, label_lengths):
              """Converts CTC labels from dense to sparse.
              Arguments:
                  labels: dense CTC labels.
                  label_lengths: length of the labels.
              Returns:
                  A sparse tensor representation of the lablels.
              """
        这里的确是说传入的是label，跟label_len, 已经手工验证过，功能的确跟上面所说的音频label的
        生成方法的功能是一致的，所以要用这个的话，要记录下每个音频label的长度
        对于keras的这个ctc_batch_cost函数，还有一点要注意的是这里：
        y_pred = math_ops.log(array_ops.transpose(y_pred, perm=[1, 0, 2]) + 1e-8)
        记得，tf的ctc_loss说：
            This class performs the softmax operation for you, so inputs should
             be e.g. linear projections of outputs by an LSTM.
        那为何，keras先对y_pred取了log？。因为softmax(log(softmax(x))) = softmax(x)
        所以，keras的ctc_batch_cost要求的y_pred，是要先经过softmax的，可以看下面这个例子
        https://ypw.io/captcha/
        以上有关keras的ctc_batch_cost,来自：
        https://stackoverflow.com/questions/43469146/keras-ctc-loss-input
    
    '''

#3月8号：
1：
    在only_keras_api中
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    ctc_loss = Lambda(ctc_lambda_func, output_shape=(1,),
                      name='ctc')([x, labels, input_length, label_length])
    
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adadelta')
    
    
    这一段compile，'ctc'指向定制的Lambda这一层，'ctc': lambda y_true, y_pred: y_pred，代表对ctc这一层的输出
    用定制的loss函数，也就是这个特殊的lambda函数，它的输入是y_true, y_pred, 然后就直接把y_pred返回，
    这样子，在model.fit阶段，keras调用这个loss函数时，会把，'ctc'这一层的输出，也就是ctc_loss，作为y_pred输入，
    得到我们所要的结果，也就是最后的loss是ctc_loss。
    而ctc_loss返回的是Tensor with shape (samples,1) containing the CTC loss of each element.正好是定制一个lossfunction
    时，keras所要求的输出：
    You can either pass the name of an existing loss function, 
    or pass a TensorFlow/Theano symbolic function that returns a scalar 
    for each data-point and takes the following two arguments:
    
    y_true: True labels. TensorFlow/Theano tensor.
    y_pred: Predictions. TensorFlow/Theano tensor of the same shape as y_true.
    The actual optimized objective is the mean of the output array across all datapoints.
    
    这样子，optimizer就会去优化ctc_loss了。
    最后关于，y_true, 应该就是model.fit的参数里面要求的y了，这个参数可以为None，可以点开看一下。
    '''

2：
    在测试use_tf_and_keras时，测试结果基本到一定次数的迭代之后，cost就下不去，而且decode出来的
    基本都是空的，不知道是空格，还是blank，可能是优化器的原因，详情见下面两篇
    http://www.tbluche.com/ctc_and_blank.html
    https://www.reddit.com/r/MachineLearning/comments/47dilt/having_issues_with_speech_recognition_using_ctc/
    https://arxiv.org/abs/1312.1737
    特别注意第一个网站，里面有一段：
    On the following figure, we show how the outputs of a network with one state per 
    character and blank evolve in the beginning of CTC training. In the top left plot, 
    we see the outputs before training. Since the weights of the network have not yet 
    been adjusted, the outputs are more or less random. As the training procedure advances, 
    the predictions of the blank symbol increases, until the posterior probability is close 
    to one for the whole sequence (end of the second line). At this point, the network 
    predicts only blank labels. Then (third line), the probabilities of character labels 
    start increasing, and peaks emerge at specific locations.
    图片可以在’相关网页离线版/Theodore Bluche.....’中看
    简而言之就是，ctc训练过程中，blank总是会先占据所有时间片，之后，字符的概率才会慢慢浮现，
    也就解释了上面的现象，而且在只用1,2,3，个句子的实验中，也总是先解码出来空的，cost都是在
    一开始迅速下降，大概cost到达30左右，后面下降的就非常缓慢，后面这个下降的阶段的速度就跟训练集
    非常有关系了！

3：
    修改了优化器为RMSPropOptimizer后，各参数如下
    bn_flag = True
    num_epochs = 20000
    num_hidden = 100
    num_layers = 1
    initial_learning_rate = 0.001
    decay_steps = 100000
    decay_rate = 1
    lstm_kernel_dropout = 0
    lstm_recurrent_dropout = 0
    batch_size = 3
    num_examples = 3 
    也就是只有固定的3个句子，大概在200个回合后，ler达到0.1，在400回合后在0.04徘徊，每回合
    耗时2.4秒，