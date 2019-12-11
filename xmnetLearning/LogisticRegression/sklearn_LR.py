"""
@ coding: UTF-8
@ Model Description：
@ Author：DuYanHui
"""
import warnings
warnings.filterwarnings("ignore")
import argparse
import csv
import os

from easydict import EasyDict as edict
import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from hdfs import Client
from sklearn.metrics import precision_score, recall_score, accuracy_score,f1_score,roc_auc_score
import time
from config import *


HDFS_HOSTS1 = "http://" + active_namenode.split(":")[0] + ":" + cfg_http_port

def train(train_path,test_path,output_path,target,
		  train_split_ratio=0.33,penalty='l2',dual=False,tol=1e-4,C=1.0,
		  random_state=None,multi_class='ovr'):
	# 设置起始时间
	time.localtime()
	time_trains_start = time.strftime('%Y{y}%m{m}%d{d} %H{h}%M{f}%S{s}'.format(y='/', m='/', d='', h=':', f=':', s=''))
	start_time = time.time()

	# 设置输入文件路径
	train_FILENAME = train_path + "/data/Data.csv"  # hdfs文件路径
	test_FILENAME = test_path + "/data/Data.csv"  # hdfs文件路径
	client = Client(HDFS_HOSTS1)
	# 训练数据读取
	with client.read(train_FILENAME) as tr_s:
		tr_content = tr_s.read()
		tr_s = str(tr_content, 'utf-8')
	# 确保文件写入完毕
	tr_file = open("trainData.csv", "w")
	tr_file.flush()
	os.fsync(tr_file)
	tr_file.write(tr_s)
	tr_file.close()
	df_train = pd.read_csv("trainData.csv", header=0)
	print(df_train)

	# 测试数据读取
	with client.read(test_FILENAME) as te_fs:
		te_content = te_fs.read()
		te_s = str(te_content, 'utf-8')
	# 确保文件写入完毕
	te_file = open("testData.csv", "w")
	te_file.flush()
	os.fsync(te_file)
	te_file.write(te_s)
	te_file.close()
	df_test = pd.read_csv("testData.csv", header=0)
	print(df_test)


	min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1,1))
	test_data_num = df_train.shape[0]
	train_data_num = df_train.shape[0]

	# 处理预测集
	df_test=min_max_scaler.fit_transform(df_test)
	df_test=np.array(df_test)

	# 数据处理和清洗
	cols = [tmp_i for tmp_i in df_train.columns if tmp_i not in [target]]
	X = df_train[cols]

	X = np.array(X)
	X = min_max_scaler.fit_transform(X)
	Y = df_train[target]
	Y = np.array(Y)

	# 训练集数据分割
	X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=train_split_ratio)

	# 使用 scikit learn 中的LR模型进行训练
	clf = LogisticRegression(penalty,dual,tol,C,random_state,multi_class,solver='liblinear')
	clf.fit(X_train,Y_train)

	# 准确率train_acc
	train_acc = clf.score(X_test,Y_test)
	print ('score Scikit learn: ', train_acc)
	# 精确率train_precision_score
	train_precision_score = precision_score(Y_test, clf.predict(X_test))
	# 召回率train_recall_score
	train_recall_score =  recall_score(Y_test, clf.predict(X_test))
	# F1_Score
	train_f1_score = f1_score(Y_test, clf.predict(X_test))
	# roc_auc_score
	train_roc_auc_score1 = roc_auc_score(Y_test, clf.predict(X_test))

	# 使用 scikit learn 中的LR模型进行预测
	result = clf.predict(df_test)
	# print(result)



	# 设置终止时间，并计算总时间
	train_end = time.time()
	train_seconds = train_end - start_time
	m, s = divmod(train_seconds, 60)
	h, m = divmod(m, 60)
	time_trains_all = "%02d:%02d:%02d" % (h, m, s)

	# ++++++++++++++++++++++++++++++++++++++++训练结果保存+++++++++++++++++++++++++++++++++++++++#
	## 保存摘要模型报告文件
	# abstract_path = HDFS_HOSTS1 + output_path + '/abstract/data/'
	abstract_path = output_path + '/abstract/data/'
	f = open('abstract.csv', mode='w', newline='')
	fileheader = ['FrameWork', 'Version', 'model', 'accuracy', 'time_trains_start', 'time_trains_all',
				  'test_data_num','train_data_num']
	w = csv.DictWriter(f, fileheader)
	w.writeheader()
	csv_dict = edict()
	csv_dict.FrameWork = 'Scikit-learn'
	csv_dict.Version = sklearn.__version__
	csv_dict.model = '%s' % LogisticRegression
	csv_dict.accuracy = str(train_acc)
	csv_dict.time_trains_start = time_trains_start
	csv_dict.time_trains_all = time_trains_all
	csv_dict.test_data_num = str(test_data_num)
	csv_dict.train_data_num = str(train_data_num)
	w.writerow(csv_dict)
	f.close()
	client.delete(abstract_path + 'abstract.csv')
	client.upload(abstract_path + 'abstract.csv','abstract.csv')
	# if len(client.list(abstract_path)):
	# 	client.delete(abstract_path + 'abstract.csv')
	# 	client.upload(abstract_path + 'abstract.csv', 'abstract.csv')
	# else:
	# 	client.upload(abstract_path + 'abstract.csv', 'abstract.csv')

	##保存模型版本信息csv文件
	version_path = output_path + '/msg/data/'
	f = open('msg.csv', mode='w', newline='')
	fileheader = ['accuracy', 'time_trains_start', 'time_trains_all', 'test_data_num', 'train_data_num']
	w = csv.DictWriter(f, fileheader)
	w.writeheader()
	csv_dict = edict()
	csv_dict.accuracy = str(train_acc)
	csv_dict.time_trains_start = time_trains_start
	csv_dict.time_trains_all = time_trains_all
	csv_dict.test_data_num = str(test_data_num)
	csv_dict.train_data_num = str(train_data_num)
	w.writerow(csv_dict)
	f.close()
	client.delete(version_path + 'msg.csv')
	client.upload( version_path + 'msg.csv','msg.csv')

	## 保存训练评价指标模型报告文件
	file_csv_path = output_path + '/evaluation/data/'
	f = open('evaluation.csv', mode='w', newline='')
	fileheader = ['accuracy', 'train_precision_score','train_recall_score','train_f1_score','train_roc_auc_score1']
	w = csv.DictWriter(f, fileheader)
	w.writeheader()
	csv_dict = edict()
	csv_dict.accuracy = str(train_acc)
	csv_dict.train_precision_score = train_precision_score
	csv_dict.train_recall_score = train_recall_score
	csv_dict.train_f1_score = train_f1_score
	csv_dict.train_roc_auc_score1 = train_roc_auc_score1
	w.writerow(csv_dict)
	f.close()
	client.delete(file_csv_path + 'evaluation.csv')
	client.upload(file_csv_path + 'evaluation.csv', 'evaluation.csv')

	# 保存测试集预测结果文件
	file_csv_path = output_path + '/result/data/'

	# 字典中的key值即为csv中列名
	dataframe = pd.DataFrame({target: result})
	# 将DataFrame存储为csv,index表示是否显示行名，default=True
	dataframe.to_csv("result.csv", index=False, sep=',')

	client.delete(file_csv_path + 'result.csv')
	client.upload(file_csv_path + 'result.csv', 'result.csv')

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='These are the parameters of the training model')


	# 正式发布用参数
	parser.add_argument('--train_path', type=str,required=True,default=None, help='Train set path')
	parser.add_argument('--test_path', type=str,required=True, default=None, help='Test set path')
	parser.add_argument('--output_path', type=str,required=True, help='The model saves the path')
	parser.add_argument('--target', type=str,required=True, default=None, help='Training set feature columns')

	parser.add_argument('--train_split_ratio', type=float, default=0.33, help='')

	parser.add_argument('--penalty', type=str, default='l2', help='model parameter')
	parser.add_argument('--dual', type=bool, default=False, help='model parameter')
	parser.add_argument('--tol', type=float, default=1e-4, help='model parameter')
	parser.add_argument('--C', type=float, default=1.0, help='model parameter')
	parser.add_argument('--random_state', type=int, default=None, help='model parameter')
	parser.add_argument('--multi_class', type=str, default='ovr', help='model parameter')

	args = parser.parse_args()  # parse_args()从指定的选项中返回一些数据

	train(train_path = args.train_path, test_path = args.test_path, output_path = args.output_path, target = args.target,
	train_split_ratio = args.train_split_ratio, penalty = args.penalty, dual = args.dual, tol = args.tol, C = args.C,
	random_state = args.random_state, multi_class = args.multi_class)