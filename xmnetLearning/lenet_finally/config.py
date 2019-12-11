import tensorflow as tf
from nets import LeNet



activate_funcs = {'ReLU': tf.nn.relu,'ReLU6':tf.nn.relu6,'Sigmoid':tf.nn.sigmoid,'Tanh':tf.nn.tanh,'eLU':tf.nn.elu,
                  'LReLU':tf.nn.leaky_relu,'Softmax':tf.nn.softmax}

loss_funcs = {'cross_entropy2': tf.nn.softmax_cross_entropy_with_logits_v2,
              'cross_entropy': tf.nn.softmax_cross_entropy_with_logits,'square_loss':tf.square}

optimizers = {'SGD':tf.train.GradientDescentOptimizer,'Adam':tf.train.AdamOptimizer}


