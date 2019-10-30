from  mxnet import ndarray

x = ndarray.arange(12)
#输出该 NDArray 对象
print(x)
#获取这个对象的形状
print(x.shape)
#得到这个对象中元素的总个数
print(x.size)
#将行向量x的形状改为(3,4)，即三行四列的矩阵
X = x.reshape((3,4))
print(X)

#输出一个张量（也就是三维矩阵），并且让矩阵中每一个元素均为零
tensor = ndarray.zeros((2,3,4))
print(tensor)

