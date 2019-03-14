（1）：
    这里用自定义的equal函数，再用Lambda包装成一个layer，好使这个几个东西”直接“返回，
    如果真的直接返回Model(output=[labels]), 会报错：labels即作为输入，又作为输出。
（2）：
    同样用Lambda包装，x要经过不属于keras.layers中的各种层的操作，都要用Lambda包装起来，
    或者自己写个类，继承keras.layers.Layer，要不然会提示AttributeError等各种错误
(3):
    K.int_shape(x)返回的是带batch_size的
(4)
    这里用了softmax，在送进tf.nn.ctc_ctc_beam_search_decoder之前，要记得手动取log，
    在tensorflow中，默认logits是没有取过softmax的，
    但是keras.backend中的ctc_batch_cost的输入要求的是取过softmax的logits，其实keras也是先对
    输入取log后，送到tf.nn.ctc_ctc_beam_search_decoder的
    这里入乡随俗，就取了softmax