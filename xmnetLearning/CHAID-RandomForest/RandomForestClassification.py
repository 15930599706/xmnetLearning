# -*- coding: utf-8 -*-
"""
@Author: DuYanHui
@Function：Random Forest（RF）+ CHAID，随机森林二分类 + 卡方特征选取
@ SampleDataSet：https://archive.ics.uci.edu/ml/machine-learning-databases/wine
"""
import argparse
import csv
import os
import time

import pandas as pd
import numpy as np
import random
import math

import sklearn
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import metrics
import collections
from pyhdfs import HdfsClient
from hdfs import Client
from easydict import EasyDict as edict
import warnings
warnings.filterwarnings("ignore")
from config import *

HDFS_HOSTS1 = "http://" + active_namenode.split(":")[0] + ":" + cfg_http_port
# 定义一棵决策树
class Tree(object):
    def __init__(self):
        self.split_feature = None
        self.split_value = None
        self.leaf_value = None
        self.tree_left = None
        self.tree_right = None

    # 通过递归决策树找到样本所属叶子节点
    def calc_predict_value(self, dataset):
        if self.leaf_value is not None:
            return self.leaf_value
        elif dataset[self.split_feature] <= self.split_value:
            return self.tree_left.calc_predict_value(dataset)
        else:
            return self.tree_right.calc_predict_value(dataset)

    # 以json形式打印决策树，方便查看树结构
    def describe_tree(self):
        if not self.tree_left and not self.tree_right:
            leaf_info = "{leaf_value:" + str(self.leaf_value) + "}"
            return leaf_info
        left_info = self.tree_left.describe_tree()
        right_info = self.tree_right.describe_tree()
        tree_structure = "{split_feature:" + str(self.split_feature) + \
                         ",split_value:" + str(self.split_value) + \
                         ",left_tree:" + left_info + \
                         ",right_tree:" + right_info + "}"
        return tree_structure


class RandomForestClassifier(object):
    def __init__(self, n_estimators=10, max_depth=-1, min_samples_split=2, min_samples_leaf=1,
                 min_split_gain=0.0, colsample_bytree="sqrt", subsample=1.0, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth if max_depth != -1 else float('inf')
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_split_gain = min_split_gain
        self.colsample_bytree = colsample_bytree  # 列采样
        self.subsample = subsample  # 行采样
        self.random_state = random_state
        self.trees = dict()
        self.feature_importances_ = dict()

    def fit(self, dataset, targets):
        assert targets.unique().__len__() == 2, "There must be two class for targets!"
        targets = targets.to_frame(name='label')
        if self.random_state:
            random.seed(self.random_state)
        random_state_stages = random.sample(range(self.n_estimators), self.n_estimators)

        # 两种列采样方式
        if self.colsample_bytree == "sqrt":
            self.colsample_bytree = int(len(dataset.columns) ** 0.5)
        elif self.colsample_bytree == "log2":
            self.colsample_bytree = int(math.log(len(dataset.columns)))
        else:
            self.colsample_bytree = len(dataset.columns)

        for stage in range(self.n_estimators):
            print(("iter: "+str(stage+1)).center(80, '='))

            # bagging方式随机选择样本和特征
            random.seed(random_state_stages[stage])
            subset_index = random.sample(range(len(dataset)), int(self.subsample * len(dataset)))
            subcol_index = random.sample(dataset.columns.tolist(), self.colsample_bytree)
            dataset_copy = dataset.loc[subset_index, subcol_index].reset_index(drop=True)
            targets_copy = targets.loc[subset_index, :].reset_index(drop=True)

            tree = self._fit(dataset_copy, targets_copy, depth=0)
            self.trees[stage] = tree
            print(tree.describe_tree())

    # 递归建立决策树
    def _fit(self, dataset, targets, depth):
        # 如果该节点的类别全都一样/样本小于分裂所需最小样本数量，则选取出现次数最多的类别。终止分裂
        if len(targets['label'].unique()) <= 1 or dataset.__len__() <= self.min_samples_split:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

        if depth < self.max_depth:
            best_split_feature, best_split_value, best_split_gain = self.choose_best_feature(dataset, targets)
            left_dataset, right_dataset, left_targets, right_targets = \
                self.split_dataset(dataset, targets, best_split_feature, best_split_value)

            tree = Tree()
            # 如果父节点分裂后，左叶子节点/右叶子节点样本小于设置的叶子节点最小样本数量，则该父节点终止分裂
            if left_dataset.__len__() <= self.min_samples_leaf or \
                    right_dataset.__len__() <= self.min_samples_leaf or \
                    best_split_gain <= self.min_split_gain:
                tree.leaf_value = self.calc_leaf_value(targets['label'])
                return tree
            else:
                # 如果分裂的时候用到该特征，则该特征的importance加1
                self.feature_importances_[best_split_feature] = \
                    self.feature_importances_.get(best_split_feature, 0) + 1

                tree.split_feature = best_split_feature
                tree.split_value = best_split_value
                tree.tree_left = self._fit(left_dataset, left_targets, depth+1)
                tree.tree_right = self._fit(right_dataset, right_targets, depth+1)
                return tree
        # 如果树的深度超过预设值，则终止分裂
        else:
            tree = Tree()
            tree.leaf_value = self.calc_leaf_value(targets['label'])
            return tree

    # 选择最好的数据集划分方式，找到最优分裂特征、分裂阈值、分裂增益
    def choose_best_feature(self, dataset, targets):
        best_split_gain = 1
        best_split_feature = None
        best_split_value = None

        for feature in dataset.columns:
            if dataset[feature].unique().__len__() <= 100:
                unique_values = sorted(dataset[feature].unique().tolist())
            # 如果该维度特征取值太多，则选择100个百分位值作为待选分裂阈值
            else:
                unique_values = np.unique([np.percentile(dataset[feature], x)
                                           for x in np.linspace(0, 100, 100)])

            # 对可能的分裂阈值求分裂增益，选取增益最大的阈值
            for split_value in unique_values:
                left_targets = targets[dataset[feature] <= split_value]
                right_targets = targets[dataset[feature] > split_value]
                split_gain = self.calc_gini(left_targets['label'], right_targets['label'])

                if split_gain < best_split_gain:
                    best_split_feature = feature
                    best_split_value = split_value
                    best_split_gain = split_gain
        return best_split_feature, best_split_value, best_split_gain

    # 选择样本中出现次数最多的类别作为叶子节点取值
    @staticmethod
    def calc_leaf_value(targets):
        label_counts = collections.Counter(targets)
        major_label = max(zip(label_counts.values(), label_counts.keys()))
        return major_label[1]

    # 分类树采用基尼指数来选择最优分裂点
    @staticmethod
    def calc_gini(left_targets, right_targets):
        split_gain = 0
        for targets in [left_targets, right_targets]:
            gini = 1
            # 统计每个类别有多少样本，然后计算gini
            label_counts = collections.Counter(targets)
            for key in label_counts:
                prob = label_counts[key] * 1.0 / len(targets)
                gini -= prob ** 2
            split_gain += len(targets) * 1.0 / (len(left_targets) + len(right_targets)) * gini
        return split_gain

    # 根据特征和阈值将样本划分成左右两份，左边小于等于阈值，右边大于阈值
    @staticmethod
    def split_dataset(dataset, targets, split_feature, split_value):
        left_dataset = dataset[dataset[split_feature] <= split_value]
        left_targets = targets[dataset[split_feature] <= split_value]
        right_dataset = dataset[dataset[split_feature] > split_value]
        right_targets = targets[dataset[split_feature] > split_value]
        return left_dataset, right_dataset, left_targets, right_targets

    # 输入样本，预测所属类别
    def predict(self, dataset):
        res = []
        for index, row in dataset.iterrows():
            pred_list = []
            # 统计每棵树的预测结果，选取出现次数最多的结果作为最终类别
            for stage, tree in self.trees.items():
                pred_list.append(tree.calc_predict_value(row))

            pred_label_counts = collections.Counter(pred_list)
            pred_label = max(zip(pred_label_counts.values(), pred_label_counts.keys()))
            res.append(pred_label[1])
        return np.array(res)

def interface(train_path,test_path,output_path,target,chaid_ratio,train_split_ratio=0.3,
              n_estimators=100,max_depth=5,min_samples_split=3,min_samples_leaf=2,min_split_gain=0.0,
              colsample_bytree="log2",subsample=0.8,random_state=100):
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


    test_data_num = df_train.shape[0]
    train_data_num = df_train.shape[0]

    # 卡方检测选出和label列最相关的前chaid_ratio（默认值为前80%）的列
    ch2 = SelectKBest(chi2, k=int(df_train.shape[1] * chaid_ratio))
    chi_df_train = pd.DataFrame(ch2.fit_transform(df_train, df_train[target]))

    label_df = df_train[target]
    # wine数据集 和 sonar 数据集
    clf = RandomForestClassifier(n_estimators,max_depth,min_samples_split,min_samples_leaf,min_split_gain,colsample_bytree,subsample,random_state)

    # 数据集分割与训练
    train_count = int(train_split_ratio* len(chi_df_train))
    clf.fit(chi_df_train.ix[:train_count],label_df.ix[:train_count])
    train_acc =  metrics.accuracy_score(label_df.ix[:train_count], clf.predict(chi_df_train.ix[:train_count]))
    print("模型的准确率：",train_acc)
    # 精确率
    train_precision_score = metrics.precision_score(label_df.ix[:train_count], clf.predict(chi_df_train.ix[:train_count]))
    # 召回率
    train_recall_score = metrics.recall_score(label_df.ix[:train_count], clf.predict(chi_df_train.ix[:train_count]))
    # F1_Score
    train_f1_score = metrics.f1_score(label_df.ix[:train_count], clf.predict(chi_df_train.ix[:train_count]))
    # roc_auc_score
    train_roc_auc_score1 = metrics.roc_auc_score(label_df.ix[:train_count], clf.predict(chi_df_train.ix[:train_count]))

    # 对测试集进行处理，保证其和训练集卡方检测后的列数一致
    ch2_list = list(ch2.get_support())
    ch2_list.pop()
    df_test_head = list(df_test.columns)
    for x, y in zip(ch2_list, df_test_head):
        if x == False :
            df_test_head.remove(y)
    df_test = df_test[df_test_head]
    # 预测
    result = clf.predict(df_test)
    # print(result)

    # 设置终止时间，并计算总时间
    train_end = time.time()
    train_seconds = train_end - start_time
    m, s = divmod(train_seconds, 60)
    h, m = divmod(m, 60)
    time_trains_all = "%02d:%02d:%02d" % (h, m, s)
    # print(time_trains_start,time_trains_all)

    # ++++++++++++++++++++++++++++++++++++++++训练结果保存+++++++++++++++++++++++++++++++++++++++#
    ## 保存摘要模型报告文件
    abstract_path = output_path + '/abstract/data/'
    f = open('abstract.csv', mode='w', newline='')
    fileheader = ['FrameWork', 'Version', 'model', 'accuracy', 'time_trains_start', 'time_trains_all',
                  'test_data_num', 'train_data_num']
    w = csv.DictWriter(f, fileheader)
    w.writeheader()
    csv_dict = edict()
    csv_dict.FrameWork = 'Scikit-learn'
    csv_dict.Version = sklearn.__version__
    csv_dict.model = '%s' % RandomForestClassifier
    csv_dict.accuracy = str(train_acc)
    csv_dict.time_trains_start = time_trains_start
    csv_dict.time_trains_all = time_trains_all
    csv_dict.test_data_num = str(test_data_num)
    csv_dict.train_data_num = str(train_data_num)
    w.writerow(csv_dict)
    f.close()
    client.delete(abstract_path + 'abstract.csv')
    client.upload(abstract_path + 'abstract.csv', 'abstract.csv')

    ##保存模型版本信息csv文件
    version_path =  output_path + '/msg/data/'
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
    client.upload(version_path + 'msg.csv', 'msg.csv')

    ## 保存训练评价指标模型报告文件
    file_csv_path =  output_path + '/evaluation/data/'
    f = open('evaluation.csv', mode='w', newline='')
    fileheader = ['accuracy', 'train_precision_score', 'train_recall_score', 'train_f1_score', 'train_roc_auc_score1']
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
    dataframe = pd.DataFrame({target: result})
    dataframe.to_csv("result.csv", index=False, sep=',')
    client.delete(file_csv_path + 'result.csv')
    client.upload(file_csv_path + 'result.csv', 'result.csv')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='These are the parameters of the training model')
    parser.add_argument('--train_path', type=str, default=None,required=True, help='Train set path')
    parser.add_argument('--test_path',  type=str,default=None,required=True, help='Test set path')
    parser.add_argument('--output_path', type=str, default=None,required=True,help='The model saves the path')
    parser.add_argument('--target', type=str,default=None,required=True, help='Training set feature columns')

    parser.add_argument('--train_split_ratio', type=float, default=0.33, help='')
    parser.add_argument('--chaid_ratio', type=float, default=0.8, help='')

    parser.add_argument('--n_estimators', type=int, default=100, help='model parameter')
    parser.add_argument('--max_depth', type=int, default=5, help='model parameter')
    parser.add_argument('--min_samples_split', type=int, default=3, help='model parameter')
    parser.add_argument('--min_samples_leaf', type=int, default=2, help='model parameter')
    parser.add_argument('--min_split_gain', type=int, default=0.0, help='model parameter')
    parser.add_argument('--colsample_bytree', type=str, default="log2", help='model parameter')
    parser.add_argument('--subsample', type=float, default=0.8, help='model parameter')

    args = parser.parse_args()  # parse_args()从指定的选项中返回一些数据

    interface(train_path = args.train_path, test_path = args.test_path, output_path = args.output_path, target = args.target,
    train_split_ratio = args.train_split_ratio,chaid_ratio = args.chaid_ratio, n_estimators = args.n_estimators,
    max_depth = args.max_depth, min_samples_split = args.min_samples_split,min_samples_leaf = args.min_samples_leaf,
    min_split_gain=args.min_split_gain ,colsample_bytree = args.colsample_bytree, subsample = args.subsample)


