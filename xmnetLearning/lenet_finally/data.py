# import pyhdfs
import tensorflow as tf
from collections import Counter
import os
import pandas as pd
import re
def get_images_labels(image_path, tag_path):
 
    # host = re.search('//(.*):',image_path).group(1)
    # client = pyhdfs.HdfsClient(hosts=host, user_name='hadoop')
    # tag_path = re.sub('hdfs://(.*?)/','/',tag_path)
    # tag = client.open(tag_path)
    # pp = pd.read_csv(tag)

    pp = pd.read_csv(tag_path)
    labels_list = []
    images_list = []
    img_path_list = []
    img_path_list = pp.values[:, 0]
    tmp = pp.values[:, 1]
    for val in tmp:
        labels_list.append(val)
    for tmp_path in img_path_list:
        images_list.append(os.path.join(image_path, tmp_path))
    image_raw = tf.gfile.FastGFile(images_list[0], 'rb').read()  # bytes
    ##image_raw = tf.gfile.FastGFile('test/a.jpg','rb').read()
    img = tf.image.decode_jpeg(image_raw)  # Tensor
    with tf.Session() as sess:
       img_size = img.eval().shape
       channels = img_size[2]
    # channels = 1
    return images_list, labels_list, len(Counter(labels_list)), channels, images_list[0][-3:]


def batch_images_labels(images_list,labels_list,batch_size,image_size,label_nums,channels,jpg_or_png):
    '''
    :param images_list: 图片地址列表，列表中的每一个元素都是一个图片的完全路径
    :param labels_list: 标签列表，与图片地址路径一一对应
    :param batch_size: 批处理数量，一次获取待训练元素的个数
    :param image_size: 图片尺寸，匹配每个模型所要求的不同规格的输入尺寸，解析图片时是必须的
    :param label_nums: 标签的类别总数，onehot编码以及网络建模时使用
    :param channels: 图片的通道数，解析图片时必须的图片属性
    :param jpg_or_png: 图片的格式，解析图片时必须的图片属性
    :return: 批处理后的图片特征和标签
    '''
    file_queue = tf.train.slice_input_producer([images_list,labels_list], shuffle=True)
    # 读取图片数据内容
    image = tf.read_file(file_queue[0])
    label = file_queue[1]
    # 解码图片数据
    if jpg_or_png == 'png':
        image = tf.image.decode_png(image,channels=channels)
    else:
        image = tf.image.decode_jpeg(image,channels=channels)
    image = tf.image.resize_image_with_crop_or_pad(image, image_size, image_size)
    # 数据标准化 训练前需要对数据进行标准化
    image = tf.image.per_image_standardization(image)
    # 批处理数据
    image_batch, label_btach = tf.train.batch([image,label], batch_size=batch_size)
    return image_batch, tf.one_hot(label_btach,label_nums)
