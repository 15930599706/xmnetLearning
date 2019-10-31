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

#创建各元素均为1的张量
print(ndarray.ones((2,3,4)))

#自定义创建NDArray
Y = ndarray.array([[1,2,3],[4,5,6],[7,8,9]])
print(Y)

#随机生成形状为（3，4）的NDArray，他的每个元素都随机采样于均值为0，标准差为1的正态分布。
print(ndarray.random.normal(0,1,shape=(2,3,4)))

#将两个NDArray做四则运算，对应位置加减乘除，并不是矩阵运算
arr1 = ndarray.array([[1,2,3],[4,5,6]])
arr2 = ndarray.array([[2,2,2],[2,2,2]])
print(arr1+arr1)
print(arr1*arr1)
print(arr1/arr2)


#求矩阵的e次幂
print(arr1.exp())
print(arr2.exp())

#矩阵乘法（A/B 等价于 A * B.T）, 【arr.T】 表示矩阵的转置，2 X 3 矩阵和 3 X 2 矩阵做乘法之后，变成 2 X 2 矩阵
print(arr1)
print(arr2.T)
print(ndarray.dot(arr1,arr2.T))

#矩阵连接，dim默认值等于1，为行连接

#行连接
print(ndarray.concat(arr1,arr2,dim=1))
#列连接
print(ndarray.concat(arr1,arr2,dim=0))

#判断矩阵值是否相等
print(arr1 == arr2)

#对NDArray中所有元素求和
print(arr1.sum())

#求矩阵的2-范数元素的平方和再开根号【欧氏距离】
print("**********")
print(arr2.norm().asscalar())


