（1）：
    这里用自定义的equal函数，再用Lambda包装成一个layer，好使这个几个东西”直接“返回，
    如果真的直接返回:Model(input=[labels], output=[labels]), 会报错：labels即作为输入，又作为输出。
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
(5)
    对ctc有关函数的输入输出总结一下
    K.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    y_true: tensor `(samples, max_label_length)`
    y_pred: tensor `(samples, time_steps, num_classes)` 要经softmax
    input_length: tensor `(samples, 1)` containing the sequence length for
            each batch item in `y_pred`.
    label_length: tensor `(samples, 1)` containing the label length for
            each batch item in `y_true`.
    return：tensor of shape(samples, 1)代表每个sample的loss
    .
    tf.nn.ctc_loss(labels, inputs, sequence_length,
                 preprocess_collapse_repeated=False,
                 ctc_merge_repeated=True,
                 ignore_longer_outputs_than_inputs=False, time_major=True)
    labels: sparsetensor, 用个sparseplaceholder, 然后向其feed用labels矩阵，和labels_length生成的稀疏矩阵
    inputs：未经softmax的logits, shape要求time_major如果time_major为真
    sequence_length: 与inputs的"timesteps"对应，如果原始输入经过卷积，要注意计算经过卷积后的timesteps长度
    time_major: 它说time_major比较快
    return：tensor of shape(samples, 1)代表每个sample的loss
    .
    tf.nn.ctc_beam_search_decoder(inputs, sequence_length, beam_width=100,
                            top_paths=1, merge_repeated=True)
    inputs:未经softmax的logits， 必须time_major, 这个函数没有time_major这个参数
    sequence_length: 与inputs的"timesteps"对应，如果原始输入经过卷积，要注意计算经过卷积后的timesteps长度
    merge_repeated=True: 这一项，对本次实验，建议改成False，如果不是连续数字， 可能merge会更好
    return：A tuple `(decoded, log_probabilities)` where
            decoded: A list of length top_paths, where `decoded[j]`
              is a `SparseTensor` containing the decoded outputs:
              `decoded[j].indices`: Indices matrix `(total_decoded_outputs[j] x 2)`
                The rows store: [batch, time].
              `decoded[j].values`: Values vector, size `(total_decoded_outputs[j])`.
                The vector stores the decoded classes for beam j.
              `decoded[j].dense_shape`: Shape vector, size `(2)`.
                The shape values are: `[batch_size, max_decoded_length[j]]`.
            log_probability: A `float` matrix `(batch_size x top_paths)` containing
                sequence log-probabilities.
    .          
    K.ctc_label_dense_to_sparse(labels, label_lengths)
    """Converts CTC labels from dense to sparse.
    .
    # Arguments
        labels: dense CTC labels.
        label_lengths: length of the labels.
    .
    # Returns
        A sparse tensor representation of the labels.
    
(6)
    
    