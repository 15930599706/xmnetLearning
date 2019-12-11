import pandas as pd
import os
from hdfs import Client
# 目前读取hdfs文件采用方式：
# 1. 先从hdfs读取二进制数据流文件
# 2. 将二进制文件另存为.csv
# 3. 使用pandas读取csv文件
HDFSHOST = "http://172.16.18.112:50070"
train_path="/9a7e7ef5a78f4f8388deff28cc5c2115/dataSet/devdatasetdef19761"
test_path="/9a7e7ef5a78f4f8388deff28cc5c2115/dataSet/devdatasetdef19762"
train_FILENAME = train_path + "/data/Data.csv" #hdfs文件路径
test_FILENAME = test_path + "/data/Data.csv" #hdfs文件路径
client = Client(HDFSHOST)
with client.read(train_FILENAME) as tr_s:
    tr_content = tr_s.read()
    tr_s = str(tr_content, 'utf-8')

# 确保文件写入完毕
tr_file = open("trainData.csv", "w")
tr_file.flush()
os.fsync(tr_file)
tr_file.write(tr_s)
tr_file.close()

# 读取文件
df_train = pd.read_csv("trainData.csv", header=0)
print(df_train)

with client.read(test_FILENAME) as te_fs:
    te_content = te_fs.read()
    te_s = str(te_content, 'utf-8')

# 确保文件写入完毕
te_file = open("testData.csv", "w")
te_file.flush()
os.fsync(te_file)
te_file.write(te_s)
te_file.close()

# 读取文件
df_test = pd.read_csv("testData.csv", header=0)
print(df_test)

# client = Client(HDFSHOST)
# tmp_path = 'http://172.16.18.112:50070/9a7e7ef5a78f4f8388deff28cc5c2115/dataSet/devdatasetdef19761/data'
# client.list(tmp_path,status='True')

