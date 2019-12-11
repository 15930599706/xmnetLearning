## 2019/10/25 by mzq

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
import numpy as np
#from PIL import Image
#from matplotlib import pyplot as plt
from pyhdfs import HdfsClient
import argparse

def inference(image_path,model_path,img_w,img_h,channel,out_result,net,**kwargs):


    client = HdfsClient(hosts='172.16.18.112,172.16.18.114', user_name='hadoop')
    active_namenode = client.get_active_namenode()
    HDFS_HOSTS = "hdfs://" + active_namenode.split(":")[0] + ":" + '9000'

    # img = Image.open(image_path)
    # image_test = np.asarray(np.resize(img, (1, img_w, img_h, channel)))

    image_path = HDFS_HOSTS + image_path + '/data/1.jpg'

    image = tf.read_file(image_path)
    image = tf.image.decode_jpeg(image, 1)
    image = tf.image.resize_image_with_crop_or_pad(image, img_w, img_h)
    image = tf.image.per_image_standardization(image)
    # image = np.asarray(np.resize(image, (1, 32,32,1)))
    image = tf.reshape(image, [1, img_w, img_h, channel])
    sess = tf.InteractiveSession()
    image_test = sess.run(image)


    hdfs_model_path = HDFS_HOSTS + model_path + '/model/'

    tf.reset_default_graph()  # 清除默认图形堆栈并重置全局默认图形。
    with tf.Session()as sess:
        meta_graph_def = tf.saved_model.loader.load(sess, [tag_constants.SERVING], hdfs_model_path)

        signature = meta_graph_def.signature_def  # 从meta_graph_def取出SignatureDef对象

        images_placeholder = signature['test_signature'].inputs['input_x'].name  # 从SignatureDef对象中找出具体的输入输出张量

        embeddinigs = signature['test_signature'].outputs['outputs'].name


        output = sess.run(embeddinigs, feed_dict={images_placeholder: image_test})
        output = sess.run(tf.nn.softmax(output))

        y = tf.argmax(output,axis=1)
        y_pred = sess.run(y)
        print(y_pred)

        # plt.imshow(img)
        # plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='These are the parameters of the training model')
    parser.add_argument('--image_path',required=True,type=str,help='')
    parser.add_argument('--model_path', required=True,type=str,help='')
    parser.add_argument('--img_w', type=int, default=32,help='')
    parser.add_argument('--img_h',type=int,default=32,help='')
    parser.add_argument('--channel', default=1, type=int,help='')
    parser.add_argument('--out_result', type=str,default=None, help='')
    #parser.add_argument('--label_file', type=str,default=None, help='')
    parser.add_argument('--net', type=str,default=None, help='')
    args = parser.parse_args()  # parse_args()从指定的选项中返回一些数据

    inference(image_path=args.image_path,model_path=args.model_path,img_h=args.img_h,img_w=args.img_w,channel=args.channel,out_result=args.out_result,net=args.net)
