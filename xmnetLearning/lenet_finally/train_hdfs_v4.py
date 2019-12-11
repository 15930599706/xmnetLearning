# 2019/10/21 by mzq

from __future__ import print_function

import argparse
import os
import sys
import random
import tensorflow as tf
from data import *
from config import *
from tensorflow.python.saved_model import tag_constants
import csv
from easydict import EasyDict as edict
import time
from pyhdfs import HdfsClient


tf.reset_default_graph() #用于清除默认图形堆栈并重置全局默认图形
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'


client = HdfsClient(hosts='172.16.18.112,172.16.18.114', user_name='hadoop')
active_namenode = client.get_active_namenode()
HDFS_HOSTS = "hdfs://" + active_namenode.split(":")[0] + ":" + '9000'


client1 = HdfsClient(hosts='172.16.18.112,172.16.18.114', user_name='hadoop')
active_namenode = client.get_active_namenode()
HDFS_HOSTS1 = "hdfs://" + active_namenode.split(":")[0] + ":" + '50070'






def get_w_b_file(loss_acc_w_b_list,file_csv):
    with open(file_csv,mode='w',newline='') as f:
        fileheader = ['train_step_i', 'train_acc','train_loss',
                      'conv1_w_men', 'conv1_w_max', 'conv1_w_min', 'conv1_b_men', 'conv1_b_max', 'conv1_b_min',
                      'conv2_w_men', 'conv2_w_max', 'conv2_w_min', 'conv2_b_men', 'conv2_b_max', 'conv2_b_min',
                      'conv3_w_men', 'conv3_w_max', 'conv3_w_min', 'conv3_b_men', 'conv3_b_max', 'conv3_b_min',
                      'fc1_w_men', 'fc1_w_max', 'fc1_w_min', 'fc1_b_men', 'fc1_b_max', 'fc1_b_min']
        w = csv.DictWriter(f, fileheader)
        w.writeheader()
        for w_b in loss_acc_w_b_list:
            line = ','.join(w_b)
            f.write(line+'\n')


def train(image_path,tag_path,output_path,label_file,train_proportion,net,activate_func,
          train_steps,batch_size,loss_func,optimizer,
          lr_rate,is_dropout=False,is_regularization=False,
          summary_output_path='summary',load_model=None,**kwargs):

    time.localtime()
    time_trains_start = time.strftime('%Y{y}%m{m}%d{d} %H{h}%M{f}%S{s}'.format(y='/', m='/', d='', h=':', f=':', s=''))
    print(time_trains_start)
    start_time = time.time()


    if net == 'LeNet':
        __size = 32
    else:
        raise Exception('Parameter error net!!')

###-------------


    # ------输入图片路径-------
    image_path = HDFS_HOSTS + image_path + "/data/"

    # ------输入标签路径-------
    tag_path = tag_path + "/data/%s" % label_file
    tag_path = client.open(tag_path)

    # -------输出模型路径------
    output_path_b = HDFS_HOSTS + output_path
###----------


    images_list, labels_list, __label_nums, __channels, jpg_or_png = get_images_labels(image_path, tag_path)

    # get data
    random.seed(521)
    random.shuffle(images_list)
    random.seed(521)
    random.shuffle(labels_list)

    slice_size = (train_proportion / 100)
    train_size_proportion = int(slice_size * len(images_list))

    train_images_list, train_labels_list = images_list[:train_size_proportion], labels_list[:train_size_proportion]
    train_data_num = len(train_images_list) + 1
    test_images_list, test_labels_list = images_list[train_size_proportion:], labels_list[train_size_proportion:]
    test_data_num = len(test_images_list)

    train_image_batch, train_label_batch = batch_images_labels(train_images_list, train_labels_list,
                                                               batch_size=batch_size, image_size=__size,
                                                               label_nums=__label_nums, channels=__channels,
                                                               jpg_or_png=jpg_or_png)
    test_image_batch, test_label_batch = batch_images_labels(test_images_list, test_labels_list,
                                                             batch_size=test_data_num, image_size=__size,
                                                             label_nums=__label_nums, channels=__channels,
                                                             jpg_or_png=jpg_or_png)

    print('Start Training model.......')

    # 设置占位符
    x_data = tf.placeholder(tf.float32, [None, __size, __size, __channels])
    y_data = tf.placeholder(tf.float32, [None, __label_nums])


    if net == 'LeNet':
        y_predict, conv1_w_men, conv1_w_max, conv1_w_min, conv1_b_men, conv1_b_max, conv1_b_min, \
        conv2_w_men, conv2_w_max, conv2_w_min, conv2_b_men, conv2_b_max, conv2_b_min, \
        conv3_w_men, conv3_w_max, conv3_w_min, conv3_b_men, conv3_b_max, conv3_b_min, \
        fc1_w_men, fc1_w_max, fc1_w_min, fc1_b_men, fc1_b_max, fc1_b_min = LeNet.model(x_data,__channels,__label_nums,activate_func,is_dropout)
    else:
        raise Exception('Network Type Error!!')


    if loss_func == 'square_loss':
        with tf.variable_scope("soft_cross"):
            loss = tf.reduce_mean(loss_funcs[loss_func](y_data - y_predict))
    else:
        with tf.variable_scope("soft_cross"):
            loss = tf.reduce_mean(loss_funcs[loss_func](labels=y_data,logits=y_predict))
    with tf.variable_scope("optimizer"):
        train_op = optimizers[optimizer](lr_rate).minimize(loss)
    with tf.variable_scope("acc"):
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.argmax(y_data, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    sess = tf.InteractiveSession()
    # 变量初始化
    sess.run(tf.global_variables_initializer())
    # 定义线程协调器和开启线程（有数据在文件当中读取提供给模型）
    coord = tf.train.Coordinator()
    # 开启线程去运行读取文件操作
    threads = tf.train.start_queue_runners(sess, coord=coord)

    train_list= []

    for i in range(train_steps):

        train_data = sess.run([train_image_batch,train_label_batch])
        train_op.run(feed_dict={x_data:train_data[0],y_data:train_data[1]})
        train_loss1 = sess.run(loss, feed_dict={x_data: train_data[0], y_data: train_data[1]})
        train_acc1 = sess.run(accuracy, feed_dict={x_data: train_data[0], y_data: train_data[1]})
        print("step:%4d;  train_loss:%f;  train_acc:%f" % (i,train_loss1,train_acc1))

        c1_w_men1, c1_w_max1, c1_w_min1, c1_b_men1, c1_b_max1, c1_b_min1, \
        c2_w_men2, c2_w_max2, c2_w_min2, c2_b_men2, c2_b_max2, c2_b_min2, \
        c3_w_men3, c3_w_max3, c3_w_min3, c3_b_men3, c3_b_max3, c3_b_min3, \
        f1_w_men4, f1_w_max4, f1_w_min4, f1_b_men4, f1_b_max4, f1_b_min4 \
            = sess.run([conv1_w_men, conv1_w_max, conv1_w_min, conv1_b_men, conv1_b_max, conv1_b_min,
        conv2_w_men, conv2_w_max, conv2_w_min, conv2_b_men, conv2_b_max, conv2_b_min,
        conv3_w_men, conv3_w_max, conv3_w_min, conv3_b_men, conv3_b_max, conv3_b_min,
        fc1_w_men, fc1_w_max, fc1_w_min, fc1_b_men, fc1_b_max, fc1_b_min])


        train_list.append(['%s' % i, '%.2f' % train_acc1, '%.2f' % train_loss1,
                           '%.3f' % c1_w_men1,'%.3f' % c1_w_max1,'%.3f' % c1_w_min1,'%.3f' % c1_b_men1,'%.3f' % c1_b_max1,'%.3f' % c1_b_min1,
                           '%.3f' % c2_w_men2,'%.3f' % c2_w_max2,'%.3f' % c2_w_min2,'%.3f' % c2_b_men2,'%.3f' % c2_b_max2,'%.3f' % c2_b_min2,
                           '%.3f' % c3_w_men3,'%.3f' % c3_w_max3,'%.3f' % c3_w_min3,'%.3f' % c3_b_men3,'%.3f' % c3_b_max3,'%.3f' % c3_b_min3,
                           '%.3f' % f1_w_men4,'%.3f' % f1_w_max4,'%.3f' % f1_w_min4,'%.3f' % f1_b_men4,'%.3f' % f1_b_max4,'%.3f' % f1_b_min4])

    test_data = sess.run([test_image_batch, test_label_batch])
    acc = sess.run(accuracy, feed_dict={x_data: test_data[0], y_data: test_data[1]})
    print("Test set accuracy:%f" % acc)

    train_end = time.time()
    train_seconds = train_end - start_time
    m, s = divmod(train_seconds, 60)
    h, m = divmod(m, 60)
    time_trains_all = "%02d:%02d:%02d" % (h, m, s)


    #client1 = HdfsClient(hosts='172.16.18.112,172.16.18.114', user_name='hadoop')
    #active_namenode = client1.get_active_namenode()
    #HDFS_HOSTS1 = "hdfs://" + active_namenode.split(":")[0] + ":" + '50070'


    ## 保存摘要模型报告文件
    abstract_path = HDFS_HOSTS1 +  output_path + '/abstract/data/'
    #if not client.exists(abstract_path):
    #    client.mkdirs(abstract_path)
    f = open('abstract.csv', mode='w', newline='')
    fileheader = ['FrameWork', 'Version', 'NetWork', 'accuracy', 'time_trains_start', 'time_trains_all', 'test_data_num',
                  'train_data_num']
    w = csv.DictWriter(f, fileheader)
    w.writeheader()
    csv_dict = edict()
    csv_dict.FrameWork = 'TensorFlow'
    csv_dict.Version = tf.__version__
    csv_dict.NetWork = '%s' % net
    csv_dict.accuracy = str(acc)
    csv_dict.time_trains_start = time_trains_start
    csv_dict.time_trains_all = time_trains_all
    csv_dict.test_data_num = str(test_data_num)
    csv_dict.train_data_num = str(train_data_num)
    w.writerow(csv_dict)
    f.close()
    client1.copy_from_local('abstract.csv',abstract_path + 'abstract.csv')
    #f.close()

    ##保存模型版本信息csv文件
    version_path = HDFS_HOSTS1 + output_path + '/msg/data/'
    #if not client.exists(version_path):
    #    client.mkdirs(version_path)
    f = open('msg.csv', mode='w', newline='')
    fileheader = ['accuracy', 'time_trains_start', 'time_trains_all','test_data_num','train_data_num']
    w = csv.DictWriter(f, fileheader)
    w.writeheader()
    csv_dict = edict()
    csv_dict.accuracy = str(acc)
    csv_dict.time_trains_start = time_trains_start
    csv_dict.time_trains_all = time_trains_all
    csv_dict.test_data_num = str(test_data_num)
    csv_dict.train_data_num = str(train_data_num)
    w.writerow(csv_dict)
    f.close()
    client1.copy_from_local('msg.csv', version_path + 'msg.csv')
    #f.close()

    ## 保存训练评价指标模型报告文件
    file_csv_path = HDFS_HOSTS1 + output_path + '/evaluation/data/'
    #if not client.exists(file_csv_path):
    #    client.mkdirs(file_csv_path)
    get_w_b_file(train_list, file_csv = 'train_list.csv')
    client1.copy_from_local('train_list.csv',file_csv_path + 'train_list.csv')



    ###----------- export saved_model -----------------
    export_path = output_path_b + "/model"
    print('Exporting trained model to ', export_path)
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)
    inputs = {'input_x': tf.saved_model.utils.build_tensor_info(x_data)}
    outputs = {'outputs': tf.saved_model.utils.build_tensor_info(y_predict)}

    signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                       method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
    builder.add_meta_graph_and_variables(sess, [tag_constants.SERVING], {'test_signature': signature})
    builder.save()

    # 回收线程
    coord.request_stop()
    coord.join(threads)





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='These are the parameters of the training model')
    parser.add_argument('--image_path',required=True,type=str,help='Picture file path')
    parser.add_argument('--tag_path',required=True,type=str,help='Note file path.')
    parser.add_argument('--output_path', required=True,type=str,help='The model saves the path')
    # parser.add_argument('--model_version',required=True,type=int,help='please input model_version,required > 0')
    parser.add_argument('--label_file', required=True, type=str, help='The label file name')
    parser.add_argument('--net',type=str,default='LeNet',help='Network model')
    parser.add_argument('--train_proportion', default=70, type=float,help='Split ratio of training set and val set')
    parser.add_argument('--activate_func',type=str,default='ReLU', help='Activation function')
    parser.add_argument('--train_steps', type=int,default=10, help='Number of iterations of test data')
    parser.add_argument('--batch_size', type=int,default=10, help='Number of batches per train')
    parser.add_argument('--loss_func', type=str,default='cross_entropy', help='loss function')
    parser.add_argument('--optimizer', type=str,default='Adam', help='Optimization function')
    parser.add_argument('--lr_rate', type=float,default=3e-4, help='learning rate')
    parser.add_argument('--is_dropout', type=bool,default=False, help='Whether to drop out')
    parser.add_argument('--is_regularization', type=bool,default=False, help='Regularization or not')
    parser.add_argument('--summary_output_path', type=str,default='summary',help='Save the path by default under the summary directory')
    parser.add_argument('--load_model', type=str,default=None, help='')
    parser.add_argument('--is_max', type=bool,default=None, help='')
    parser.add_argument('--is_min', type=bool,default=None, help='')
    parser.add_argument('--is_mean', type=bool,default=None, help='')
    parser.add_argument('--is_std', type=bool,default=None, help='')
    parser.add_argument('--is_cross_entropy', type=bool,default=None, help='')
    parser.add_argument('--is_accuracy', type=bool,default=None, help='')

    parser.add_argument('--url1',type=str,default=None,help='')
    parser.add_argument('--url2', type=str, default=None, help='')
    parser.add_argument('--pyName', type=str, default=None, help='')
    parser.add_argument('--cpu', type=int, default=None, help='')
    parser.add_argument('--gpu', type=int, default=None, help='')
    parser.add_argument('--memory', type=str, default=None, help='')
    parser.add_argument('--executor', type=str, default=None, help='')
    parser.add_argument('--projectId', type=str, default=None, help='')
    parser.add_argument('--guid', type=str, default=None, help='')
    parser.add_argument('--token', type=str, default=None, help='')
    parser.add_argument('--extend_params', type=str, default=None, help='')


    parser.add_argument('--slice_type', type=str, default=None, help='')
    parser.add_argument('--engine', type=str, default=None, help='')
    parser.add_argument('--height', type=int, default=None, help='')
    parser.add_argument('--test_proportion', type=str, default=None, help='')
    parser.add_argument('--evaluate_target', type=str, default=None, help='')
    parser.add_argument('--width', type=int, default=None, help='')
    parser.add_argument('--net_optimize', type=str, default=None, help='')
    parser.add_argument('--engine_version', type=str, default=None, help='')
    args = parser.parse_args()  # parse_args()从指定的选项中返回一些数据



    train(image_path=args.image_path,tag_path=args.tag_path,output_path=args.output_path,label_file=args.label_file,
          train_proportion=args.train_proportion,net=args.net,activate_func=args.activate_func,train_steps=args.train_steps,
          batch_size=args.batch_size,loss_func=args.loss_func,optimizer=args.optimizer,
          lr_rate=args.lr_rate,is_dropout=args.is_dropout,is_regularization=args.is_regularization,
          summary_output_path=args.summary_output_path,
          load_model=args.load_model)


