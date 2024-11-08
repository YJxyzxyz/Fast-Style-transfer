import tensorflow as tf, pdb

# 用于初始化网络权重的标准差，设置为0.1。
WEIGHTS_INIT_STDEV = .1


# 该函数接受一个图像作为输入，并通过多个卷积层、残差块和反卷积层进行处理。
# 最终，通过tanh激活函数将输出限制在[-1, 1]范围内，并线性变换到[0, 255]范围，这可能是为了与图像像素值的常见范围相匹配。
def net(image):
    conv1 = _conv_layer(image, 32, 9, 1)
    conv2 = _conv_layer(conv1, 64, 3, 2)
    conv3 = _conv_layer(conv2, 128, 3, 2)
    resid1 = _residual_block(conv3, 3)
    resid2 = _residual_block(resid1, 3)
    resid3 = _residual_block(resid2, 3)
    resid4 = _residual_block(resid3, 3)
    resid5 = _residual_block(resid4, 3)
    conv_t1 = _conv_tranpose_layer(resid5, 64, 3, 2)
    conv_t2 = _conv_tranpose_layer(conv_t1, 32, 3, 2)
    conv_t3 = _conv_layer(conv_t2, 3, 9, 1, relu=False)
    preds = tf.nn.tanh(conv_t3) * 150 + 255. / 2
    return preds


# 定义一个标准的卷积层，包括卷积操作、实例归一化和ReLU激活函数（可选）。
# num_filters：卷积核的数量。
# filter_size：卷积核的大小。
# strides：卷积步长。
# relu：是否在应用卷积后使用ReLU激活函数。
def _conv_layer(net, num_filters, filter_size, strides, relu=True):
    weights_init = _conv_init_vars(net, num_filters, filter_size)
    strides_shape = [1, strides, strides, 1]
    net = tf.nn.conv2d(net, weights_init, strides_shape, padding='SAME')
    net = _instance_norm(net)
    if relu:
        net = tf.nn.relu(net)

    return net


# 定义一个反卷积（或称为转置卷积）层，通常用于上采样或扩大特征图的尺寸。
# 该函数与_conv_layer类似，但使用的是tf.nn.conv2d_transpose进行反卷积操作。
def _conv_tranpose_layer(net, num_filters, filter_size, strides):
    weights_init = _conv_init_vars(net, num_filters, filter_size, transpose=True)

    batch_size, rows, cols, in_channels = [i.value for i in net.get_shape()]
    new_rows, new_cols = int(rows * strides), int(cols * strides)
    # new_shape = #tf.pack([tf.shape(net)[0], new_rows, new_cols, num_filters])

    new_shape = [batch_size, new_rows, new_cols, num_filters]
    tf_shape = tf.stack(new_shape)
    strides_shape = [1, strides, strides, 1]

    net = tf.nn.conv2d_transpose(net, weights_init, tf_shape, strides_shape, padding='SAME')
    net = _instance_norm(net)
    return tf.nn.relu(net)

# 定义一个残差块，它包含两个卷积层，并将输入（即“残差”）添加到第二个卷积层的输出上。
# 这种结构有助于网络学习恒等映射，从而更容易地进行优化，并可能提高性能。
def _residual_block(net, filter_size=3):
    tmp = _conv_layer(net, 128, filter_size, 1)
    return net + _conv_layer(tmp, 128, filter_size, 1, relu=False)

# 实现实例归一化（Instance Normalization），这是一种在风格迁移等任务中常用的归一化技术。
# 它对每个样本的每个通道分别进行归一化，与批量归一化不同，它不依赖于批次中的其他样本。
def _instance_norm(net, train=True):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)
    shift = tf.Variable(tf.zeros(var_shape))
    scale = tf.Variable(tf.ones(var_shape))
    epsilon = 1e-3
    normalized = (net - mu) / (sigma_sq + epsilon) ** (.5)
    return scale * normalized + shift

# 用于初始化卷积层的权重。
# 权重使用截断的正态分布进行初始化，标准差由WEIGHTS_INIT_STDEV定义。
def _conv_init_vars(net, out_channels, filter_size, transpose=False):
    _, rows, cols, in_channels = [i.value for i in net.get_shape()]
    if not transpose:
        weights_shape = [filter_size, filter_size, in_channels, out_channels]
    else:
        weights_shape = [filter_size, filter_size, out_channels, in_channels]

    weights_init = tf.Variable(tf.truncated_normal(weights_shape, stddev=WEIGHTS_INIT_STDEV, seed=1), dtype=tf.float32)
    return weights_init
