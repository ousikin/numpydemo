import numpy as np
import numpy.random as npr

print(np.__version__)

weight = np.array([0.1, 0.2, 0.3, 0.4])
print(type(weight))
print(weight.shape)

stock_return = np.array([
    [0.003731, 0.021006, 0.004854, 0.40056, 0.5489],
    [0.1889, 0.2456, 0.3556, 0.46778, 0.56680],
    [-0.122334, -0.24566, -0.3, 0.48899, 0.5444556],
    [-0.024112, 0.011704, 0.029563, -0.01457, 0.01629]

]
)
print(stock_return)
print(stock_return.shape)

print("------------------------------------")
weight_list = [0.1, 0.2, 0.3, 0.4]
weight_array = np.array(weight_list)
print(weight_array)

return_list = [0.003731, 0.021006, 0.004854, 0.40056, 0.5489,
               0.1889, 0.2456, 0.3556, 0.46778, 0.56680,
               -0.122334, -0.24566, -0.3, 0.48899, 0.5444556,
               -0.024112, 0.011704, 0.029563, -0.01457, 0.01629]

return_array = np.array(return_list)
return_array = return_array.reshape(4, 5)  # reshape将数组升唯
print(return_array)

print("-------------------------------------")
# ravel将数组降唯
return_array1 = return_array.ravel()
print(return_array1)

print("-------------------------------------")
print(return_array.ndim)
print(weight_array.ndim)

print(return_array.size)
print(weight_array.size)

print(return_array.itemsize)
print(weight_array.itemsize)

print(return_array.dtype)
print(weight_array.dtype)

print(np.float64(True))
print(np.float64(False))

print(np.int8(67.0))
print(np.uint8(78.0))

print(np.float32(78))

# 数组的便捷生成
print("---------------------")

a = np.arange(10)

print(a)

b = np.arange(1, 20, 2)
print(b)

c = np.arange(1.0, 1.9, 0.1)

print(np.linspace(0, 100, 51))  # 生成等差数列

print(np.logspace(start=0, stop=2, num=5))
print(np.logspace(start=2, stop=5, num=5, base=2, endpoint=False))

zero_array1 = np.zeros(5)
print(zero_array1)

zero_array2 = np.zeros((4, 5))
print(zero_array2)

print("===============================")

zero_weight = np.zeros_like(weight_array)
print(zero_array1)

zero_weight2 = np.zeros_like(return_array1)
print(zero_weight2)

print("====================================")

one_weight = np.ones(4)
print(one_weight)

one_weight1 = np.ones((4, 5))
print(one_weight1)

print("=================================")
print("已成功了一大半了")
one_weight3 = np.ones_like(zero_array2)
print(one_weight3)

print("===========单位矩阵==================")
i = np.eye(5)
print(i)

print("==========上=================")
a = np.eye(4, k=1)
print(a)

print("===============下================")
a = np.eye(4, k=-1)
print(a)

a = np.eye(4, k=-3)
print(a)

print("=============one-hot===================")
labels = np.array([[1], [2], [0], [1]])

a = np.eye(3)[1]
print("类别好是1,转成one-hot的形式:", a, "\n")

a = np.eye(3)[2]
print("类别号是2,转成one-hot的形式:", a, "\n")

# a = np.eye(3)[1, 0]
# print("1转成one-hot的形式,数组的第一个数字为:", a, "\n")
#
# a = np.eye()[[1, 2, 0, 1]]
# print("类别号为1, 2, 0, 1 one-hot的形式,数组的第一个数字为:\n", a)

# res = np.eye(3)[labels.reshape(-1)]
# print("label转成one-hot形式的结果:\n", res, "\n")
# print("labels转成one-hot后的大小:", res.shape)

print("---------------创建方单位矩阵-----------------")
print(np.identity(3))

print(return_array[1, 2])
print(weight_array[-2])
print(weight_array[[0, 1, 3]])

print(np.where(return_array < -0.01))

print(return_array[2:, 1:4])
print(return_array[1])
print(return_array[:, 2])

print("----------sort排序----------------")
print("-------axis=0按列排序----------------")
print(np.sort(return_array, axis=0))
print("-------axis=1按行排序----------------")
print(np.sort(return_array, axis=1))
print("-------axis=0按行排序----------------")
print(np.sort(return_array))

print("----------sum用于数组求和----------------")
print(return_array.sum(axis=0))

print("------------sum(axis=1)用于按行求和-------------")
print(return_array.sum(axis=1))

print("----------sum用于数组中所有元素求和----------------")
print(return_array.sum())

print("---------------prod求乘积-----------------------------")

print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

print("----------max(axis=1)用于按列求最值----------------")
print(return_array.max(axis=0))

print("------------max(axis=1)用于按行求值-------------")
print(return_array.max(axis=1))

print("----------max用于数组中所有元素求值----------------")
print(return_array.max())

print("----------求平均数----------------------------")
print(return_array.mean(axis=0))
print(return_array.mean(axis=1))
print(return_array.mean())

print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
print("----------求方差----------------------------")
print(return_array.var(axis=0))
print(return_array.var(axis=1))
print(return_array.var())

print("----------求标准差----------------------------")
print(return_array.std(axis=0))
print(return_array.std(axis=1))
print(return_array.std())

print(np.sqrt(return_array))  # 开方
print(np.square(return_array))  # 平方
print(np.exp(return_array))  # e为底的指数次方

print("-------------------------------------------------")
print(np.log(return_array))
print(np.log10(return_array))
print(np.log2(return_array))

print("==============================数组间的运算=============================================")
print(return_array + 1)
print(return_array - 1)
print(return_array * 1)
print(return_array / 1)
print(return_array ** 1)

print("------------------------------------------------------")
return_max = np.maximum(return_array, zero_array1)
print(return_max)
print("==========================================================")
return_min = np.minimum(return_array, zero_array1)
print(return_min)

print('----------------------------相关系系数的矩阵------------------------------------')
corrcoef_return = np.corrcoef(return_array)
print(corrcoef_return)

a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 2, 3, 4, 5])

print(np.dot(a, b))

print("**********************************randome********************************************************")
x_norm = npr.normal(loc=1.0, scale=2.0, size=10000)
print("正态分布中抽取平均值", x_norm.mean())
print("正态分布中抽取标准差", x_norm.std())

x_snorm1 = npr.rand(10000)
x_snorm2 = npr.standard_normal(size=10000)
x_snorm3 = npr.normal(loc=0, scale=1.0, size=10000)

x_logn = npr.lognormal(mean=0.5, sigma=1.0, size=10000)
print('从对数分布中抽样的平均值', x_logn.mean())
print('从对数分布中抽样的标准差', x_logn.std())

x_chi1 = npr.chisquare(df=4, size=10000)
x_chi2 = npr.chisquare(df=100, size=10000)
print('从自由度等于4的卡方分布中的抽样的平均值', x_chi1.mean())
print('从自由度等于4的卡方分布中的抽样的标准差', x_chi1.std())
print('从自由度等于100的卡方分布中的抽样的平均值', x_chi2.mean())
print('从自由度等于100的卡方分布中的抽样的标准差', x_chi2.std())

x_t1 = npr.standard_t(df=2, size=10000)
x_t2 = npr.standard_t(df=120, size=10000)
print('从自由度等于2的学生t分布中的抽样的平均值', x_t1.mean())
print('从自由度等于2的学生t分布中的抽样的标准差', x_t1.std())
print('从自由度等于100的学生t分布中的抽样的平均值', x_t2.mean())
print('从自由度等于100的学生t分布中的抽样的标准差', x_t2.std())
