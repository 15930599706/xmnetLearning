import tensorflow as tf
import tensorflow as tf
from pyhdfs import HdfsClient
# 端口配置
cfg_http_port = '50070'
cfg_rpc_port = '9000'
user_name='hadoop'
hosts='172.16.18.112,172.16.18.114'
# 数据集接口url
url1 = 'http://test.cop.com/vbap3/dsc/dataSetDefs/updateByOther'
url2 = 'http://test.cop.com/vbap3/dsc/dataSetDefs/uploadFields'
# hdfs配置
client = HdfsClient(hosts=hosts, user_name=user_name)
active_namenode = client.get_active_namenode()
HDFS_HOSTS = "hdfs://" + active_namenode.split(":")[0] + ":" + cfg_rpc_port

client1 = HdfsClient(hosts=hosts, user_name=user_name)
active_namenode = client.get_active_namenode()
HDFS_HOSTS1 = "hdfs://" + active_namenode.split(":")[0] + ":" + cfg_http_port








