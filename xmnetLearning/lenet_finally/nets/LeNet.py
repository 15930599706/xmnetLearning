from config import *
activate_funcs = {'ReLU': tf.nn.relu,'ReLU6':tf.nn.relu6,'Sigmoid':tf.nn.sigmoid,'Tanh':tf.nn.tanh,'eLU':tf.nn.elu,
                  'LReLU':tf.nn.leaky_relu,'Softmax':tf.nn.softmax}

def model(image,channels,label_nums,activate_func,is_dropout):
    '''
    :param image: 网络模型的输入数据
    :param label_nums: 网络模型的输出类别数目
    :param activate_func: 激活函数
    :return: 网络模型的预测值
    '''
    with tf.name_scope('conv1'):
        conv1_w = tf.Variable(tf.truncated_normal(shape=[5, 5, channels, 6], mean=0, stddev=0.1))
        conv1_b = tf.Variable(tf.zeros(6))
        conv1 = activate_funcs[activate_func](tf.nn.conv2d(image, conv1_w, strides=[1, 1, 1, 1], padding='VALID') + conv1_b)
    # S2 Pooling Input=28*28*6 Output=14*14*6
    with tf.name_scope('pool1'):
        pool_1 = activate_funcs[activate_func](
            tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
    # C3 conv Input=14*14*6 Output=10*10*16
    with tf.name_scope('conv2'):
        conv2_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 6, 16], mean=0, stddev=0.1))
        conv2_b = tf.Variable(tf.zeros(16))
        conv2 = activate_funcs[activate_func](
            tf.nn.conv2d(pool_1, conv2_w, strides=[1, 1, 1, 1], padding='VALID') + conv2_b)
    # S4 Pooling Input=10*10*16 OutPut=5*5*16
    with tf.name_scope('pool2'):
        pool_2 = activate_funcs[activate_func](tf.nn.avg_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'))
    # C5 Input=5*5*16 Output=1*1*120
    with tf.name_scope('conv3'):
        conv3_w = tf.Variable(tf.truncated_normal(shape=[5, 5, 16, 120], mean=0, stddev=0.1))
        conv3_b = tf.Variable(tf.zeros(120))
        conv3 = activate_funcs[activate_func](
            tf.nn.conv2d(pool_2, conv3_w, strides=[1, 1, 1, 1], padding='VALID') + conv3_b)
    with tf.name_scope('fc1'):
        fc = tf.reshape(conv3, (-1, 120))
        # F6 Input=120 OutPut=84
        fc1_w = tf.Variable(tf.truncated_normal(shape=(120, 84), mean=0, stddev=0.1))
        fc1_b = tf.Variable(tf.zeros(84))
        fc1 = activate_funcs[activate_func](tf.matmul(fc, fc1_w) + fc1_b)

    if is_dropout:
        keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # placeholder是占位符
        fc1 = tf.nn.dropout(fc1, keep_prob)


    with tf.name_scope('fc2'):
        # F7 Input=84  Output=10
        fc2_w = tf.Variable(tf.truncated_normal(shape=(84, label_nums), mean=0, stddev=0.1))
        fc2_b = tf.Variable(tf.zeros(label_nums))
    y_predict = tf.matmul(fc1, fc2_w) + fc2_b

    conv1_w_men = tf.reduce_mean(conv1_w)
    conv1_w_max = tf.reduce_max(conv1_w)
    conv1_w_min = tf.reduce_min(conv1_w)

    conv1_b_men = tf.reduce_mean(conv1_b)
    conv1_b_max = tf.reduce_max(conv1_b)
    conv1_b_min = tf.reduce_min(conv1_b)

    conv2_w_men = tf.reduce_mean(conv2_w)
    conv2_w_max = tf.reduce_max(conv2_w)
    conv2_w_min = tf.reduce_min(conv2_w)

    conv2_b_men = tf.reduce_mean(conv2_b)
    conv2_b_max = tf.reduce_max(conv2_b)
    conv2_b_min = tf.reduce_min(conv2_b)

    conv3_w_men = tf.reduce_mean(conv3_w)
    conv3_w_max = tf.reduce_max(conv3_w)
    conv3_w_min = tf.reduce_min(conv3_w)

    conv3_b_men = tf.reduce_mean(conv3_b)
    conv3_b_max = tf.reduce_max(conv3_b)
    conv3_b_min = tf.reduce_min(conv3_b)

    fc1_w_men = tf.reduce_mean(fc1_w)
    fc1_w_max = tf.reduce_max(fc1_w)
    fc1_w_min = tf.reduce_min(fc1_w)

    fc1_b_men = tf.reduce_mean(fc1_b)
    fc1_b_max = tf.reduce_max(fc1_b)
    fc1_b_min = tf.reduce_min(fc1_b)



    return y_predict,conv1_w_men,conv1_w_max,conv1_w_min,conv1_b_men,conv1_b_max,conv1_b_min,\
           conv2_w_men,conv2_w_max,conv2_w_min,conv2_b_men,conv2_b_max,conv2_b_min,\
           conv3_w_men, conv3_w_max, conv3_w_min, conv3_b_men, conv3_b_max, conv3_b_min,\
           fc1_w_men,fc1_w_max,fc1_w_min,fc1_b_men,fc1_b_max,fc1_b_min
