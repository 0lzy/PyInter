import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt

#array数组
a0=np.array([[1,2,3,4],[1,2,3,4]])
print(a0)

a=np.array([1,2,3,4],ndmin=2)
print(a)
#ndmin为数组中的最小维数

b=np.array([1,2,3,4],dtype=complex)
#dtype为数组中数据类型
print(b)

#dtype数据类型
# numpy.dtype(object, align, copy)
# Object：被转换为数据类型的对象。
# Align：如果为true，则向字段添加间隔，使其类似 C 的结构体。
# Copy : 生成dtype对象的新副本，如果为flase，结果是内建数据类型对象的引用。
#以下示例定义名为 student 的结构化数据类型，其中包含字符串字段name，整数字段age和浮点字段marks。 此dtype应用于ndarray对象。
#int8，int16，int32，int64 可替换为等价的字符串 'i1'，'i2'，'i4'，以及其他。
dt = np.dtype([('name','S20'), ('age', np.int16),('marks','f4')])
c = np.array([('abc', 21, 50), ('xyz', 18, 75)], dtype=dt)
print(c)
# 文件名称可用于访问 age 列的内容
print(c['name'])

#数组属性
d = np.array([[1,2,3],[4,5,6]])
#返回一个包含数组维数元祖
print(d.shape)
print(d)
#改变数组维数
d.shape=(3,2) #直接改变原数组维数
d1=d.reshape(3,2)  #定义一个新的数组来接收改变维数后的数组
print(d.shape)
print(d)
print(d1)

#numpy.itemsize这一数组属性返回数组中每个元素的字节单位长度。
x = np.array([1,2,3,4,5], dtype = np.int8)
print(x.itemsize)
x = np.array([1,2,3,4,5], dtype = np.float32)
print(x.itemsize)  

#np.array对象属性,numpy.flags这个函数返回了它们的当前值。没什么卵用
# 1.	C_CONTIGUOUS (C) 数组位于单一的、C 风格的连续区段内
# 2.	F_CONTIGUOUS (F) 数组位于单一的、Fortran 风格的连续区段内
# 3.	OWNDATA (O) 数组的内存从其它对象处借用
# 4.	WRITEABLE (W) 数据区域可写入。 将它设置为flase会锁定数据，使其只读
# 5.	ALIGNED (A) 数据和任何元素会为硬件适当对齐
# 6.	UPDATEIFCOPY (U) 这个数组是另一数组的副本。当这个数组释放时，源数组会由这个数组中的元素更新
print(x.flags)

# numpy.empty,它创建指定形状和dtype的未初始化数组。
#numpy.empty(shape, dtype = float, order = 'C')
#Order 'C'为按行的 C 风格数组，'F'为按列的 Fortran 风格数组
#shape为数组形状，整数或整数元组类型
f= np.empty([3,2], dtype = int,order='F') 
print(f)
#np.empty创建的一个未初始化的数组，数组中元素为随机值
#np。zeros可返回以特定值0填充的数组
f1=np.zeros([3,2],dtype=np.int_)
f2=np.zeros(5,dtype=np.int,order='C')
print(f1)
print(f2)
#np.ones可返回一个以特定值1填充的数组
f3=np.ones([3,1],dtype=np.int32)
print(f3)

#numpy.asarray此函数类似于numpy.array，除了它有较少的参数。 这个例程对于将 Python 序列(不是数组的元素类型：元组，列表等)转换为np.array（数组类型或矩阵）非常有用。
# numpy.asarray(a, dtype = None, order = None)
g=[1,2,3] #这个是列表
print(g)
g1=np.asarray(g)
print(g1)
# 来自元组的 ndarray  
g=(1,2,3)
g1=np.asarray(g)
print(g1)
# 来自元组列表的 ndarray
x =  [(1,2,3),(4,5)] 
a = np.asarray(x)  
print(a) 

#frombuffer将data以流的形式读入转化成ndarray对象
#numpy.frombuffer(buffer, dtype = float, count = -1, offset = 0) 
#默认从0位置开始(offset=0)，读取所有数据(count=-1)
#data是字符串的时候，Python3默认str是Unicode类型，所以要转成bytestring在原str前加上b
s =  b'Hello World' 
a = np.frombuffer(s, dtype =  'S1')  
print(a)

# 使用 range 函数创建列表对象   
list = range(5)  #左闭右开
print(list)
print(np.asarray(list).flags)

#numpy.fromiter,此函数从任何可迭代对象构建一个ndarray对象，返回一个新的一维数组。
# 从列表中获得迭代器  
list = range(5) 
it = iter(list)  
# 使用迭代器创建 ndarray 
#迭代器是一个可以记住遍历的位置的对象。
#迭代器对象从集合的第一个元素开始访问，直到所有的元素被访问完结束。迭代器只能往前不会后退。
for i in range(3):
    next(it)
x = np.fromiter(it, dtype =  float) 
#numpy.fromiter,此函数从任何可迭代对象构建一个ndarray对象，返回一个新的一维数组。 
#这里迭代器迭代到了第三个位置，还剩下两个位置
print(x)

#arange生成等距一维数组，默认从0开始，步距为1，不包含最后一个
e = np.arange(24) 
print(e)
#返回数组维度
print(e.ndim)
#虽然只能生成一维数组，但是可以通过reshape改变数组维数
e1 = e.reshape(2,4,3) 
print(e1)
print(e1.ndim)
x = np.arange(10,20,2,dtype=np.float32)
print(x) 

#numpy.linspace
#此函数类似于arange()函数。 在此函数中，指定了范围之间的均匀间隔数量，而不是步长。
x = np.linspace(10,20,5)  #要在10到20之间等距生成5个数，左闭右闭(endpoint=true)  
print(x)
x = np.linspace(10,20,5, endpoint =  False) #序列中是否包含stop值，默认为ture
print(x)
#retstep 如果为true，返回样例，以及连续数字之间的步长
x = np.linspace(10,20,5, retstep=True,dtype=np.int32)
print(x)

#numpy.logscale(start, stop, num, endpoint, base, dtype)
#此函数返回一个ndarray对象，其中包含在对数刻度上均匀分布的数字。 刻度的开始和结束端点是某个底数的幂，通常为 10。
#num生成均匀分布数字的数量，base为底数（默认为10）
a = np.logspace(1.0,  2.0, num =  10)  
print(a)
# 将对数空间的底数设置为 2  
a = np.logspace(1,10,num=10,base=2,dtype=np.int32)  
print(a)

#基本切片
#slice对象被传递给数组来提取数组的一部分。
a = np.arange(10)
print(a)
s = slice(2,7,2)  #不包含7
print(a[s])
#或者
b=a[2:7:2]
print(b)
#如果只输入一个参数，则将返回与索引对应的单个项目。
#如果使用a:，则从该索引向后的所有项目将被提取。
# 如果使用两个参数（以:分隔），则对两个索引（不包括停止索引）之间的元素以默认步骤进行切片。
a = np.arange(10)
b = a[5]  
print(b)
a = np.arange(10)  
print (a[2:])
a = np.arange(10)  
print (a[2:5])
#多维
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print(a)  
print (a[1:])
#切片还可以包括省略号（...），来使选择元组的长度与数组的维度相同。 如果在行位置使用省略号，它将返回包含列中元素的ndarray。
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print(a)  
# 这会返回第二列元素的数组：  
print(a[...,1])  
# 现在我们从第二行切片所有元素：   
print(a[1,...])    
# 现在我们从第二列向后切片所有元素：
print(a[...,1:])

#高级索引,返回数组任意位置确切值

#整数索引，每个整数数组表示该维度的下标值。
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print(a[[0,1,2],[0,1,2]]) 
#该结果包括数组中(0,0)，(1,1)和(2,2)位置处的元素
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])   
print(x)  
print(x.ndim)
rows = np.array([[0,0],[3,3]]) 
print(rows)
cols = np.array([[0,2],[0,2]])
print(cols) 
y = x[rows,cols]    
y1=x[[[0,0],[3,3]],[[0,2],[0,2]]]  #多维返回数组各个角处值 
print(x[[0,0,3,3],[0,2,0,2]])  #一维返回数组的各个角处值
print('这个数组的每个角处的元素是：')  
print(y)
print(y1)
z = x[1:4,1:3]     #左闭右开
print ('切片之后，我们的数组变为：' ) 
print(z)
#若用高级索引
z=x[1:4,[1,2]]
import numpy as np 
x = np.array([[  0,  1,  2],[  3,  4,  5],[  6,  7,  8],[  9,  10,  11]])  
print(x[x > 5])
# 什么时候numpy中会出现nan？
# 当我们读取本地的文件为float的时候，如果有缺失，就会出现nan
# 当做了一个不合适的计算的时候 (比如无穷大(inf)减去无穷大)
a = np.array([np.nan,1,2,np.nan,3,4,5])  
a1=a[~np.isnan(a)]
print(a1)
a = np.array([1,2+6j,5,3.5+5j])
#过滤非复数元素  
print(a[np.iscomplex(a)])

#NumPy - 广播是指 NumPy 在算术运算期间处理不同形状的数组的能力。
# 对数组的算术运算通常在相应的元素上进行。 如果两个阵列具有完全相同的形状，则这些操作被无缝执行。
a = np.array([1,2,3,4],order='C') 
print(a)
b = np.array([10,20,30,40],order='F')
print(b) 
c = a * b 
print(c)
#如果两个数组的维数不相同，则元素到元素的操作是不可能的。
# 然而，在 NumPy 中仍然可以对形状不相似的数组进行操作，因为它拥有广播功能。 
# 较小的数组会广播到较大数组的大小，以便使它们的形状可兼容。
# 如果满足以下规则，可以进行广播：
# ndim较小的数组会在前面追加一个长度为 1 的维度。
# 输出数组的每个维度的大小是输入数组该维度大小的最大值。
# 如果输入在每个维度中的大小与输出大小匹配，或其值正好为 1，则在计算中可它。
# 如果输入的某个维度大小为 1，则该维度中的第一个数据元素将用于该维度的所有计算。
a = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0], [30.0, 30.0, 30.0]])
b = np.array([1.0, 2.0, 3.0])
print(a)
print(b)
print(a+b)
print(a.ndim)
print(b.ndim)

#NumPy 包包含一个迭代器对象numpy.nditer。
# 它是一个有效的多维迭代器对象，可以用于在数组上进行迭代。
# 数组的每个元素可使用 Python 的标准Iterator接口来访问。
a = np.arange(0, 60, 5)
a = a.reshape(3, 4) #改变一维数组a的维度
print(a)
for i in np.nditer(a):
    print(i)
for i in a:
    print(i)
#迭代顺序
#迭代的顺序匹配数组的内容布局，而不考虑特定的排序。 这可以通过迭代上述数组的转置来看到。
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print(a) 
b = a.T   #a的转置
print(b) 
for x in np.nditer(b):  
    print(x)
#如果相同元素使用 F 风格（列）顺序存储，则迭代器选择以更有效的方式对数组进行迭代。
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print(a)
b = a.T 
print(b)
c = b.copy(order='C')   #以行风格重写
print(c)
for x in np.nditer(c):   #以行风格迭代
    print(x)  
c = b.copy(order='F')    #以列风格重写
print(c)
for x in np.nditer(c):   #以列风格迭代
    print(x)
#可以通过显式提醒，来强制nditer对象使用某种顺序：
a = np.arange(0,60,5).reshape(3,4)
print(a)
for i in np.nditer(a,order='C'):
    print(i)
for i in np.nditer(a,order='F'):
    print(i)
#nditer对象有另一个可选参数op_flags。
# 其默认值为只读，但可以设置为读写或只写模式。这将允许使用此迭代器修改数组元素。
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print(a)
for x in np.nditer(a,op_flags=['readwrite']):  #读写模式 
    x[...]=2*x 
print(a)
# nditer类的构造器拥有flags参数，它可以接受下列值：
# 1.c_index 可以跟踪 C 顺序的索引
# 2.f_index 可以跟踪 Fortran 顺序的索引
# 3.multi-index 每次迭代可以跟踪一种索引类型
# 4.external_loop 给出的值是具有多个值的一维数组，而不是零维数组
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print(a)
for x in np.nditer(a,flags=['external_loop'], order = 'F'):  
    print(x)
#广播迭代
#如果两个数组是可广播的，nditer组合对象能够同时迭代它们。
# 假设数组a具有维度 3X4，并且存在维度为 1X4 的另一个数组b，则使用以下类型的迭代器（数组b被广播到a的大小）。
a = np.arange(0,60,5) 
a = a.reshape(3,4)  
print(a)   
b = np.array([1,  2,  3,  4],dtype=int)  
print(b)  
for x,y in np.nditer([a,b]):  
    print(x,y)

#数组操作
#numpy.ndarray.flat该函数返回数组上的一维迭代器，行为类似 Python 内建的迭代器。 
a = np.arange(8).reshape(2,4) 
print(a)
# 返回展开数组中的下标的对应元素 
print(a.flat[5])
# numpy.ndarray.flatten该函数返回折叠为一维的数组副本，函数接受下列参数：(order)
#区别于nditer中的参数flags
#order：'C' -- 按行，'F' -- 按列，'A' -- 原顺序，'k' -- 元素在内存中的出现顺序。
a = np.arange(8).reshape(2,4) 
print(a)
print(a.flatten()) 
print(a.flatten(order = 'F'))
#numpy.ravel这个函数返回展开的一维数组，并且按需生成副本。返回的数组和输入数组拥有相同数据类型。这个函数接受两个参数。 
a = np.arange(8).reshape(2,4) 
print(a)
print(a.ravel())  
print(a.ravel(order = 'F'))
#np.ravel ()和np.flatten ()的区别 两者的功能是一致的，将多维数组降为一维，但是两者的区别是返回拷贝还是返回视图，
# np.flatten (0返回一份拷贝，对拷贝所做修改不会影响原始矩阵，而np.ravel ()返回的是视图，修改时会影响原始矩阵

#翻转操作
#numpy.transpose这个函数翻转给定数组的维度。如果可能的话它会返回一个视图。函数接受下列参数
#numpy.transpose(arr, axes)
#arr：要转置的数组
#axes：整数的列表，对应维度，通常所有维度都会翻转。
a = np.arange(12).reshape(3,4) 
print(a) 
print(np.transpose(a))
#numpy.ndarray.T该函数属于ndarray类，行为类似于numpy.transpose。
print(a.T)
#numpy.rollaxis该函数向后滚动特定的轴，直到一个特定位置。这个函数接受三个参数：
#arr：输入数组
#axis：要向后滚动的轴，其它轴的相对位置不会改变
#start：默认为零，表示完整的滚动。会滚动到特定位置。
# 创建了三维的 ndarray 
a = np.arange(8).reshape(2,2,2) 
print(a)
# 将轴 2 滚动到轴 0（宽度到深度）
print(np.rollaxis(a,2))  
# 将轴 0 滚动到轴 1：（宽度到高度） 
print(np.rollaxis(a,2,1))
#在滚动之后的新坐标系中，对于坐标 (x,y,z)，其对应的位置为 (z,x,y)。
#numpy.swapaxes该函数交换数组的两个轴。对于 1.10 之前的 NumPy 版本，会返回交换后数组的试图。这个函数接受下列参数：
# 创建了三维的 ndarray 
a = np.arange(8).reshape(2,2,2) 
print(a) 
# 现在交换轴 0（深度方向）到轴 2（宽度方向）
print(np.swapaxes(a, 2, 0))

#修改维度
#broadcast如前所述，NumPy 已经内置了对广播的支持。
# 此功能模仿广播机制。 它返回一个对象，该对象封装了将一个数组广播到另一个数组的结果。
x = np.array([[1], [2], [3]]) 
y = np.array([4, 5, 6])  
print(y)
print(x)
# 对 y 广播 x
b = np.broadcast(x,y)
r,c = b.iters 
print(r.__next__)
#它拥有 iterator 属性，基于自身组件的迭代器元组  
# shape 属性返回广播对象的形状
print(b.shape)
# 手动使用 broadcast 将 x 与 y 相加 
#创建一个和b相同维度的为初始化的数组，其中数据随机生成
c = np.empty(b.shape,dtype=np.int32)
print(c)
print(c.shape)
c.flat=[u + v for (u,v) in b]
print(c)
# 获得了和 NumPy 内建的广播支持相同的结果
print(x + y)

#numpy.broadcast_to此函数将数组广播到新形状。 它在原始数组上返回只读视图。
# 它通常不连续。 如果新形状不符合 NumPy 的广播规则，该函数可能会抛出ValueError。
a = np.arange(4).reshape(1,4)    #一行四列
print(a)  
print(np.broadcast_to(a,(4,4)))

x = np.array(([1,2],[3,4])) 
print(x)   
y = np.expand_dims(x, axis = 0) 
print(y)
print(x.shape, y.shape) 
# # 在位置 1 插入轴
y = np.expand_dims(x, axis = 1)   
print(y)
print(y.shape)
print(x.ndim,y.ndim) 

#numpy.squeeze函数从给定数组的形状中删除一维条目.
x = np.arange(9).reshape(1,3,3) 
print(x)  
y = np.squeeze(x,axis=0) 
print(y)
print(x.shape, y.shape)

#数组的连接
#numpy.concatenate数组的连接是指连接。 
#此函数用于沿指定轴连接相同形状的两个或多个数组。 该函数接受以下参数。
import numpy as np 
a = np.array([[1,2],[3,4]]) 
print(a)
b = np.array([[5,6],[7,8]]) 
print(b)  
# 两个数组的维度相同 
#沿轴 0 连接两个数组
print(np.concatenate((a,b)))   #axis默认为0，
#沿轴 1 连接两个数组 
print(np.concatenate((a,b),axis = 1))

#numpy.stack此函数沿新轴连接数组序列。
a = np.array([[1, 2], [3, 4]])
print(a)
b = np.array([[5, 6], [7, 8]])
print(b)
print(a.shape)
#沿轴 0 堆叠两个数组：
print(np.concatenate((a,b),axis=0))
#区别于concatenate:在原来的轴上连接数组
print(np.stack((a, b), 0))
#沿轴 1 堆叠两个数组：
c=np.stack((a, b), 1)
print(c)
print(c.shape)

#numpy.hstack是numpy.stack函数的变体，通过堆叠来生成水平的单个数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
#水平堆叠：
c = np.hstack((a, b))
print(c)

#numpy.vstack是numpy.stack函数的变体，通过堆叠来生成竖直的单个数组
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
c=np.vstack((a,b))
print(c)

#数组分割
#numpy.split(ary, indices_or_sections, axis)
# indices_or_sections：可以是整数，表明要从输入数组创建的，等大小的子数组的数量。
# 如果此参数是一维数组，则其元素表明要创建新子数组的点。
a = np.arange(9)
print(a)
#'将数组分为三个大小相等的子数组：'
b = np.split(a, 3)
print(b)
#'将数组在一维数组中表明的位置分割：'
b = np.split(a, [4,6])
#分割出数组4到5的部分
print(b)

#numpy.hsplit是split()函数的特例，其中轴为 1 表示水平分割，无论输入数组的维度是什么。
a = np.arange(16).reshape(4, 4)
print(a)
#'水平分割：'
b = np.hsplit(a,2)
#其中2为参数indices_or_sections
print(b)

#numpy.vsplit是split()函数的特例，其中轴为 0 表示竖直分割，无论输入数组的维度是什么
a=np.arange(16).reshape(4,4)
b=np.vsplit(a,2)
print(b)

#添加/删除元素
#numpy.resize此函数返回指定大小的新数组。 如果新大小大于原始大小，则包含原始数组中的元素的重复副本。
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.shape)
b = np.resize(a,(3, 2))
#类似于
c=np.reshape(a,(3,2))
print(c)
print(b)
print(b.shape)
# 要注意 a 的第一行在 b 中重复出现，因为尺寸变大了
b = np.resize(a, (3, 3))
print(b)

#numpy.append此函数在输入数组的末尾添加值。
#如果没有提供插入数组，则插入之后的数组将被展开
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a)
print(a.shape)
#'向数组添加元素：'
print(np.append(a, [7, 8, 9]))
#'沿轴 0 添加元素：'
print(a.shape)
# 附加操作不是原地的，而是分配新的数组。 此外，输入数组的维度必须匹配否则将生成ValueError。
print(np.append(a, [[7, 8, 9]], axis=0))
#'沿轴 1 添加元素：'
print(np.append(a, [[5, 5, 5], [7, 8, 9]], axis=1))

#numpy.append()函数适用于添加单个元素，而numpy.insert()函数适用于插入多个元素。
# 在大多数情况下，使用numpy.append()函数更为常见，因为它更简单且更安全，可以避免一些常见的错误。
# numpy.insert(arr, obj, values, axis)
# arr：输入数组
# obj：在其之前插入值的索引
# values：要插入的值
# axis：沿着它插入的轴，如果未提供，则输入数组会被展开
a = np.array([[1, 2], [3, 4], [5, 6]])
print(a)
print(a.shape)
#'未传递 Axis 参数。 在插入之前输入数组会被展开。'
print(np.insert(a, 3, [11, 12]))  #3为插入位置
# '传递了 Axis 参数。 会广播值数组来配输入数组。'
# '沿轴 0 广播：'
#要插入的数组维度小于原数组才会被广播
print(np.insert(a, 1, [11], axis=0))
print(np.insert(a,1,[11,12],axis=0))
b=11
# '沿轴 1 广播：'
print(np.insert(a, 1, 11, axis=1))
#print(np.stack((a,b),axis=1))
#这个stack只能用来指定多个数组延指定的轴连接
#而不能想insert一样来插入

#numpy.unique此函数返回输入数组中的去重元素数组。 
# 该函数能够返回一个元组，包含去重数组和相关索引的数组。
#  索引的性质取决于函数调用中返回参数的类型。
a = np.array([5,2,6,2,7,5,6,8,2,9]) 
#针对于一维数组，如果不是一维数组则会展开
#'第一个数组：' 
print(a)
#'第一个数组的去重值：' 
u = np.unique(a) 
print(u)

# numpy.unique(arr, return_index, return_inverse, return_counts)
# arr：输入数组，如果不是一维数组则会展开
# return_index：如果为true，返回输入数组中的元素下标
# return_inverse：如果为true，返回原数组中每个元素在唯一值数组中的索引。这样，我们可以通过返回数组的操作重构原始数组。
# return_counts：如果为true，返回去重数组中的元素在原数组中的出现次数
#'去重数组的索引数组：' 
u = np.unique(a, return_index = True)
#返回去重后数组所含的剩余元素以及其在原数组中的下标
#分别返回一个一维数组
print(u)
print(a)
#这里再次查看原数组发现原数组已经被修改，可见unique是直接作用于原数组的，而不是返回一个新的数组
u,indices= np.unique(a,return_inverse = True)
#返回去重后的数组和返回原数组中每个元素在唯一值数组中的索引
#分别返回一个一维数组
#这里注意u对应的是去重后的数组，而indices对应的是原数组元素在去重数组中的下标
print(u) 
#'使用下标重构原数组：' 
print(u[indices]) 
#'返回去重元素的重复数量：'(以数组形式) 
u= np.unique(a,return_counts = True) 
print(u) 

# bin()是Python内置的一个函数，用于将十进制整数转换为二进制字符串。
# 它接收一个整数参数n，并返回一个以“0b”开头的二进制字符串，该字符串表示了n的二进制表达形式。例如：
# >>> bin(10)
# '0b1010'
# 如果要将这个二进制字符串转换回整数形式，可以使用int()函数，并指定第二个参数为2，以表示它是一个二进制字符串。例如：
# >>> int('0b1010', 2)
# 10

# 在进行位与运算时，只有当两个操作数的对应位都为1时，结果的对应位才为1。
# 以下是一个具体的例子，假设我们要对两个二进制数1110和1010进行位与运算：
# 1110
# & 1010
# -------
#   1010
# 可以看到，在上面的例子中，只有1110和1010每个位置上都是1的位置（第二位和第四位）才会得到结果中相应位置上的1，因此最终结果是1010。
# 可以这样理解：位与运算就像是比较两个二进制数的每一位，只有当它们都是1的时候就会返回1，否则就会返回0。
# 这相当于是实现了对两个二进制数同时“开关”的效果。如果两个二进制数的某一个位置上是0，就相当于把“开关”打开了，不会取到结果；
# 只有两个二进制数的每一个位置上都是1，才会相当于是把两个“开关”都打开了，才会取得相应的结果。
# 因此，只有当两个二进制数的对应位都是1时，对应位置上的结果才为1，否则就为0。

#位操作
#通过np.bitwise_and()函数对输入数组中的整数的二进制表示的相应位执行位与运算。
#'13 和 17 的二进制形式：' 
a,b = 13,17 
print(bin(a),bin(b)) 
#'13 和 17 的位与运算，返回十进制数
print(np.bitwise_and(13, 17))
#当两个二进制数进行位与运算时，如果产生的结果中至少有一位是1，那么我们通常会说这个位与运算结果为1。
# 这是因为在二进制数的运算中，1通常表示“真”或“成立”，而0表示“假”或“不成立”，
# 因此如果结果中至少有一个地方是“真”，那么整个位与运算的结果就可以看作是“真”或“成立”。

#通过np.bitwise_or()函数对输入数组中的整数的二进制表示的相应位执行位或运算。
a,b = 13,17 
#'13 和 17 的二进制形式：' 
print(bin(a), bin(b)) 
# '13 和 17 的位或运算，返回十进制数
print(np.bitwise_or(13,17))

#invert此函数计算输入数组中整数的位非结果。 对于有符号整数，返回补码。
#'13 的位反转，其中 ndarray 的 dtype 是 uint8：' 
#uint8 是一种无符号 8 位整数类型
#13的二进制表示是1101，
# 因为在无符号的8位整数范围内，13的二进制表示前面使用了4个0作为填充，这样可以得到：00001101
print(np.invert(np.array([13], dtype = np.uint8))) 
#比较 13 和 242 的二进制表示，我们发现了位的反转 
#binariy_repr查看对应的二进制表示方法，width限制长度
print(np.binary_repr(13, width = 8)) 
print(np.binary_repr(242, width = 8))

#numpy.left shift()函数将数组元素的二进制表示中的位向左移动到指定位置，右侧附加相等数量的 0。
#'将 10 左移两位：' 
print(bin(10))  #前面有0b
print(np.binary_repr(10))  #前面没有0b
print(np.left_shift(10,2)) 
print(np.binary_repr(40))

#numpy.right_shift()函数将数组元素的二进制表示中的位向右移动到指定位置，左侧附加相等数量的 0。
print(np.right_shift(40,2)) 
print(np.binary_repr(40, width = 8))
print(np.binary_repr(10, width = 8))
#  '00001010' 中的两位移动到了右边，并在左边添加了两个 0。

#NumPy - 字符串函数
#	add() 返回两个str或Unicode数组的逐个字符串连接
print(np.char.add(['hello'],[' xyz']))
print(np.char.add(['hello', 'hi'],[' abc', ' xyz']))

#multiply() 返回按元素多重连接后的字符串 
print(np.char.multiply('Hello ',3))

# center() 返回给定字符串的副本，其中元素位于特定字符串的中央，填充元素在两侧使其长度，满足参数要求
# np.char.center(arr, width,fillchar) 
print(np.char.center('hello',20,fillchar = '*'))

# capitalize() 返回给定字符串的副本，其中只有第一个字符串大写
print(np.char.capitalize('hello world'))

# title() 返回字符串或 Unicode 的按元素标题转换版本
#使每一个字符的第一位大写
print(np.char.title('hello how are you?'))

# lower() 返回一个数组，其元素转换为小写
print(np.char.lower(['HELLO','WORLD']))
print(np.char.lower('HELLO'))

# upper() 返回一个数组，其元素转换为大写
print(np.char.upper('hello') )
print(np.char.upper(['hello','world'])
      )
# split() 返回字符串中的单词列表，并使用分隔符来分割
# 默认情况下，空格用作分隔符。 否则，指定的分隔符字符用于分割字符串。
print(np.char.split ('hello how are you?')) 
print(np.char.split ('TutorialsPoint Hyderabad Telangana', sep = 'd'))

# splitlines() 返回元素中的行列表，以换行符分割
#'\n'，'\r'，'\r\n'都会用作换行符。
print(np.char.splitlines('hello\nhow are you?'))
print(np.char.splitlines('hello\rhow are you?'))

# strip() 返回数组副本，其中元素移除了开头或者结尾处的特定字符
print(np.char.strip('ashok arora','a'))
print(np.char.strip(['arora','admin','java'],'a'))

# join() 返回一个字符串，它是序列中字符串的连接
print(np.char.join(':','dmy'))
print(np.char.join([':','-'],['dmy','ymd']))
# replace() 返回字符串的副本，其中所有子字符串的出现位置都被新字符串取代
print(np.char.replace ('He is a good boy', 'is', 'was'))

# decode() 按元素调用str.decode
a = np.char.encode('hello', 'cp500') 
print (a)
print(np.char.decode(a,'cp500'))

# encode() 按元素调用str.encode
a = np.char.encode('hello', 'cp500') 
print(a)

#算数运算
#用于执行算术运算（如add()，subtract()，multiply()和divide()）的输入数组必须具有相同的形状或符合数组广播规则。
a = np.arange(9, dtype = np.float_).reshape(3,3)  
print(a)
#相同形状位置元素元素或广播后对应位置运算
b = np.array([10,10,10])  
print(b)
print(np.add(a,b))  #+
print(a+b)
print(np.subtract(a,b))   #-
print(np.multiply(a,b))   #*
print(np.divide(a,b))    #/

#numpy.reciprocal()此函数返回参数逐元素的倒数。
#由于 Python 处理整数除法的方式，对于绝对值大于 1 的整数元素，结果始终为 0， 对于整数 0，则发出溢出
a = np.array([0.25,  1.33,  1,  0,  100])  
print(a)
print(np.reciprocal(a))  
b = np.array([100], dtype =  int)    
print(b) 
print(np.reciprocal(b))  

#numpy.power()此函数将第一个输入数组中的元素作为底数，计算它以第二个输入参数为指数的数组中相应元素的幂，并返回一个新的数组 
a = np.array([10,100,1000])  
print(a)    
print(np.power(a,2))   
b = np.array([1,2,3])  
print(b)  
#第二个参数也可以是一个数组
print(np.power(a,b))

#numpy.mod()此函数返回输入数组中相应元素的除法余数。
a = np.array([10,20,30]) 
b = np.array([3,5,7])  
print(a)  
print(b) 
print(np.mod(a,b))  
# 函数numpy.remainder()也产生相同的结果。
print(np.remainder(a,b))

#算数函数
#三角函数
#NumPy 拥有标准的三角函数，它为弧度制单位的给定角度返回三角函数比值。
a = np.array([0,30,45,60,90])  
# 通过乘 pi/180 转化为弧度  
print(np.sin(a*np.pi/180))  
#'数组中角度的余弦值：'  
print(np.cos(a*np.pi/180))   
#'数组中角度的正切值：'  
print(np.tan(a*np.pi/180))
#arcsin，arccos，和arctan函数返回给定角度的sin，cos和tan的反三角函数
#以上方法只能转换为弧度制
# 这些函数的结果可以通过numpy.degrees()函数通过将弧度制转换为角度制来验证。
a = np.array([0,30,45,60,90])  
# '含有正弦值的数组：'
sin = np.sin(a*np.pi/180)  
print(sin)
# '计算角度的反正弦，返回值以弧度为单位：'
inv = np.arcsin(sin)  
print(inv)
#'通过转化为角度制来检查结果：'  
print(np.degrees(inv))  
#'arccos 和 arctan 函数行为类似：'
cos = np.cos(a*np.pi/180)  
print(cos)
# '反余弦：'
inv = np.arccos(cos)    
print(inv)
# '角度制单位：'  
print(np.degrees(inv))  
# 'tan 函数：'
tan = np.tan(a*np.pi/180)  
print(tan)
#'反正切：'
inv = np.arctan(tan)  
print(inv)
#'角度制单位：'  
print(np.degrees(inv)) 

#舍入函数
#numpy.around()这个函数返回四舍五入到所需精度的值。 该函数接受以下参数。
#numpy.around(a,decimals)
# decimals 要舍入的小数位数。 默认值为0。 如果为负，整数将四舍五入到小数点左侧的位置
a = np.array([1.0,5.55,  123,  0.567,  25.532])  
print(a)
#'舍入后：'  
print(np.around(a)) 
print(np.around(a, decimals =  1))  
print(np.around(a, decimals =  -1))

#numpy.floor()此函数返回不大于输入参数的最大整数
#注意在Python中，向下取整总是从 0 舍入。
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])  
print(a)
print(np.floor(a))

#numpy.ceil()函数返回输入值的上限，即，标量x的上限是最小的整数i ，使得i> = x。
a = np.array([-1.7,  1.5,  -0.2,  0.6,  10])  
print(np.ceil(a))

#统计NumPy 有很多有用的统计函数，用于从数组中给定的元素中查找最小，最大，百分标准差和方差等。
#numpy.amin() 和 numpy.amax()
#这些函数从给定数组中的元素沿指定轴返回最小值和最大值。
a = np.array([[3,7,5],[8,4,3],[2,4,9]])  
print(a)
#注意这里不能简单地理解为这种操作按行或列的方向进行计算
#1为在数组的第2维度上进行计算
#在这里就指的是行方向
print(np.amin(a,1))
#同理如下
print(np.amin(a,0))
print(np.amax(a,axis=1))
print(np.amax(a,axis=0))

#numpy.ptp()函数返回沿轴的值的范围（最大值 - 最小值）。
import numpy as np 
a = np.array([[3,7,5],[8,4,3],[2,4,9]])  
print(a)
print(np.ptp(a,axis=1))  #第一个位置为7-4
print(np.ptp(a,axis=0))
#无参数的情况下，会将元素组展开为1维数组再进行计算
#amin和amax同理
print(np.ptp(a))

#第 p% 个数的位置 = （n+1）× p% / 100
#numpy.percentile(a, q, axis)百分位数是统计中使用的度量，表示小于这个值得观察值占某个百分比
#q 要计算的百分位数，在 0 ~ 100 之间
a = np.array([[30,40,70],[80,20,10],[50,90,60]])  
print(a)
print(np.percentile(a,50))
#计算方法：先找到从小到大排序，再找到百分数位的位置（要求有q%的数据比它小）
#没有第三个参数则展开为1维6数组
print(np.percentile(a,50,1))
print(np.percentile(a,50,0))

#numpy.median()中值定义为将数据样本的上半部分与下半部分分开的值。
#中位数
a = np.array([[30,65,70],[80,95,10],[50,90,60]])
print(a)
print(np.median(a))
print(np.median(a,axis=0))
print(np.median(a,axis=1)) 

# numpy.mean()
# 算术平均值是沿轴的元素的总和除以元素的数量。 
# numpy.mean()函数返回数组中元素的算术平均值
# 如果提供了轴，则沿其计算。
a = np.array([[1,2,3],[3,4,5],[4,5,6]])  
print(a)
print(np.mean(a))
print(np.mean(a,axis=0))
print(np.mean(a,axis=1))

#numpy.average()可计算加权平均值
a = np.array([1,2,3,4])
print(a)
wts=np.array([4,3,2,1])
print(np.average(a))
#以上不加权情况
print(np.average(a,weights=wts))
#加权平均值 = (1*4+2*3+3*2+4*1)/(4+3+2+1)
# 如果 returned 参数设为 true，则返回权重的和  
print(np.average(a,weights=wts,returned=True))
a = np.arange(6).reshape(3,2) 
#指定数组的轴
wt=np.array([3,5])
print(np.average(a,axis=1,weights=wt,returned=True))

# std = sqrt(mean((x - x.mean())**2))
#标准差是与均值的偏差的平方的平均值的平方根。
a=np.array([1,2,3,4])
print(np.std(a))

#方差是偏差的平方的平均值，即mean((x - x.mean())** 2)。
#标准差是方差的平方根。
print(np.var(a))

#NumPy - 排序、搜索和计数函数
# 种类	                   速度	    时间复杂度	    空间复杂度	稳定性
# 'quicksort'（快速排序）	1	     O(n^2)	           0	     否
# 'mergesort'（归并排序）	2	     O(n*log(n))	  ~n/2	     是
# 'heapsort'（堆排序）	    3	     O(n*log(n))	   0	     否

# numpy.sort(a, axis, kind, order)
# 1.	a 要排序的数组
# 2.	axis 沿着它排序数组的轴，如果没有数组会被展开，沿着最后的轴排序
# 3.	kind 默认为'quicksort'（快速排序）
# 4.	order 如果数组包含字段，则是要排序的字段
a=np.array([[3,7],[9,1]])
print(a)
print(np.sort(a))  #在数组的最后一个轴（每一行上）由小到大排序
#以上也就是axis=1
print(np.sort(a,axis=1))  #可见和没有给出轴的情况下输出相同
print(np.sort(a,axis=0))   #在列上排序

#给出一个结构体类型
dt=np.dtype([('name','S10'),('age',int)])
a = np.array([("raju",21),("anil",25),("ravi",17),("amar",27)], dtype = dt)
print(a)
print(np.sort(a,order='name'))  #按名字的字符大小排序
#递归排序
print(np.sort(a,kind='mergesort',order='age'))
#堆排序
print(np.sort(a,kind='heapsort',order='age'))

#numpy.argsort()函数对输入数组沿给定轴执行间接排序，并使用指定排序类型返回数据的索引数组。 
#这个索引数组用于构造排序后的数组。
x = np.array([3,  1,  2])  
print(x)
print(np.argsort(x))
#以上返回排序后的在原数组中的索引数组
print(x[np.argsort(x)])
y=np.argsort(x)
for i in y:
    print(x[i])
#将迭代器重构为数组
print(np.fromiter(x[y], dtype = int)) 

#numpy.lexsort()这个没看懂

#numpy.argmax() 和 numpy.argmin()
#这两个函数分别沿给定轴返回最大和最小元素的索引。
a = np.array([[30,40,70],[80,20,10],[50,90,60]]) 
print(a)
print(np.argmax(a))
#展开数组
# print(a.flatten())
maxindex = np.argmax(a, axis =  0) 
print(maxindex)
maxindex = np.argmax(a, axis =  1) 
print(maxindex)
minindex = np.argmin(a) 
print(minindex)
minindex=np.argmin(a,axis=0)
print(minindex)
minindex=np.argmin(a,axis=1)
print(minindex)

#minindex = np.argmin(a) 
a = np.array([[30,40,0],[0,20,10],[50,0,60]])  
print(a)
print(np.nonzero(a))

#where()函数返回输入数组中满足给定条件的元素的索引。
x=np.arange(9).reshape(3,3)
print(x)
print(np.where(x==1))
#使用索引获取满足条件的元素
print(x[np.where(x==1)])
print(np.where(x>3))

#extract()函数返回满足任何条件的元素。
#注意用途相同，但where是返回索引
x=np.arange(9).reshape(3,3)
print(x)
condition=np.mod(x,2)==0
#mod为取模运算
print(condition) 
print(np.extract(condition,x)) 
print(np.where(np.mod(x,2)==0))

# a=np.arange(6)
# #简单的赋值不会创建数组对象的副本。 相反，它使用原始数组的相同id()来访问它。
# # id()返回 Python 对象的通用标识符，类似于 C 中的指针。
# #此外，一个数组的任何变化都反映在另一个数组上。 例如，一个数组的形状改变也会改变另一个数组的形状。

# #map()
# def square(x):
#     return  pow(x,2)
# a=np.arange(6)
# b=list(map(square,[1,2,3,4,5]))
# #在python3中map返回的是一个对象
# #这里要用list转换，否则会出现乱码报错
# print(b)
# #通过使用lambda匿名函数的方法使用map()函数：
# x=[1,3,5,7,9]
# y=[2,4,6,8,10]
# print(list(map(lambda x,y:x+y,x,y)))
# #通过lambda函数使返回值是一个元组：
# print(list(map(lambda x,y:(x**y,x+y),x,y)))
# #当不传入function时，map()就等同于zip()，将多个列表相同位置的元素归并到一个元组：

#存储在计算机内存中的数据取决于 CPU 使用的架构。
# 它可以是小端（最小有效位存储在最小地址中）或大端（最小有效字节存储在最大地址中）
#numpy.ndarray.byteswap()函数在两个表示：大端和小端之间切换。
a=np.array([1,256,8755],dtype=np.int16)
print(a)
#以16进制表示内存中的数据
print(map(hex,a))
print(a.byteswap(True))
print(map(hex,a))

# 当内容物理存储在另一个位置时，称为副本。
# 另一方面，如果提供了相同内存内容的不同视图，我们将其称为视图。
a=np.arange(6)
print(a)
#它使用原始数组的相同id()来访问它。 id()返回 Python 对象的通用标识符，类似于 C 中的指针。
print(id(a))
b=a
print(b)
#简单的赋值不会创建数组对象的副本。
print(id(b))
#这里可以看到没有创建副本
b.shape=(2,3)
#修改了b的形状之后，a的形状也改变了
print(b)
print(a)

#ndarray.view()方法，它是一个新的数组对象，并可查看原始数组的相同数据。 
# 与前一种情况不同，新数组的维数更改不会更改原始数据的维数。
a=np.arange(6).reshape(3,2)
print(a)
print(id(a))
#创建a的视图
b=a.view()
print(b)
#这里是相同的内容存储在了另一个位置
print(id(b))
#修改b的形状，但不会修改a
b.shape=(2,3)
print(b)
print(a)

#数组的切片也会创建视图：
a = np.array([[10,10],  [2,3],  [4,5]])  
print(a)
s = a[:,:2]
print(id(a))
print(s)  
print(id(s))

#深复制
#ndarray.copy()函数创建一个深层副本。 它是数组及其数据的完整副本，不与原始数组共享。
a=np.array([[10,10],[2,3],[4,5]])
print(a)
b=a.copy()
print(b)
#b与a不共享任何内容
print(b is a)
b[0,0]=100
print(b)
print(a)
#修改b的内容后，a不变
c=a.view()
print(c)
#浅复制也不共享
print(c is a)

#NumPy 包包含一个 Matrix库numpy.matlib。此模块的函数返回矩阵而不是返回ndarray对象。
print(np.empty((2,2)))
#返回一个随机生成的矩阵
#matlib.empty()函数返回一个新的矩阵，而不初始化元素matlib.empty()函数返回一个新的矩阵，而不初始化元素
a=np.matlib.empty((2,2))
print(a)

#numpy.matlib.zeros()此函数返回以零填充的矩阵。
print(np.matlib.zeros((2,2)))
print(np.zeros((2,2)))
#我只能说我没看出来这几个的区别

#numpy.matlib.ones()此函数返回以一填充的矩阵。
print(np.matlib.ones((2,2),dtype=int))

#numpy.matlib.eye()这个函数返回一个矩阵，对角线元素为 1，其他位置为零。
#numpy.matlib.eye((n,M,k,dtype))
#n为行数，m为列数，默认为n
#k为对角线的索引，（从第一行的第几个数开始，注意以0为起点）
x=np.matlib.eye(3,4,3,dtype=np.int16)
print(x)

#numpy.matlib.identity()函数返回给定大小的单位矩阵。单位矩阵是主对角线元素都为 1 的方阵。
x=np.matlib.identity(3,dtype=int)
#这个只能生成方阵
print(x)

#numpy.matlib.rand()`函数返回给定大小的填充随机值的矩阵。
print(np.matlib.rand(3,3))

#注意，矩阵总是二维的，而ndarray是一个 n 维数组。 两个对象都是可互换的。
i=np.matrix('1,2;3,4')
print(i)
j=np.asarray(i)
print(j)
k=np.asmatrix(j)
print(k)

#线性代数
#numpy.dot()
#此函数返回两个数组的点积。 对于二维向量，其等效于矩阵乘法。
# 对于 N 维数组，它是a的最后一个轴上的和与b的倒数第二个轴的乘积。
a=np.array([[1,2],[3,4]])
b=np.array(([[11,12],[13,14]]))
c=np.array([1,2])
d=np.array([3,4])
# 对于一维数组，它是向量的内积。
print(np.dot(c,d))
print(a)
print(b)
print(np.dot(a,b))

#numpy.vdot()此函数返回两个向量的点积。
# 如果第一个参数是复数，那么它的共轭复数会用于计算。 如果参数id是多维数组，它会被展开。
a=np.array([[1,2],[3,4]])
b=np.array(([[11,12],[13,14]]))
print(np.vdot(a,b))
#1*11 + 2*12 + 3*13 + 4*14 = 130

#numpy.inner()此函数返回一维数组的向量内积。 对于更高的维度，它返回最后一个轴上的和的乘积。
c=np.array([1,2])
d=np.array([3,4])
print(np.inner(c,d))
a=np.array([[1,2],[3,4]])
b=np.array(([[11,12],[13,14]]))
print(a)
print(b)
print(np.inner(a,b))
#以上参数为多维数组时：
#1*11+2*12 1*13+2*14
#3*11+4*12 3*13+4*14

#numpy.matmul()函数返回两个数组的矩阵乘积。
#虽然它返回二维数组的正常乘积，但如果任一参数的维数大于2，则将其视为存在于最后两个索引的矩阵的栈，并进行相应广播。
#另一方面，如果任一参数是一维数组，则通过在其维度上附加 1 来将其提升为矩阵，并在乘法之后被去除。
a=np.array([[1,2],[3,4]])
b=np.array(([[11,12],[13,14]]))
print(np.matmul(a,b))
#二维与一维
a=np.array([[1,0],[0,1]])
b=np.array([1,2])
print(np.matmul(a,b))
print(np.matmul(b,a))
#纬度大于二的数组
a=np.arange(8).reshape(2,2,2)
b=np.arange(4).reshape(2,2)
print(np.matmul(a,b))

#numpy.linalg.det()行列式在线性代数中是非常有用的值。
# 它从方阵的对角元素计算。 对于 2×2 矩阵，它是左上和右下元素的乘积与其他两个的乘积的差。
#[[a，b]，[c，d]]，行列式计算为ad-bc
a=np.array([[1,2],[3,4]])
#行列式的计算
print(int(np.linalg.det(a)))
b=np.array([[6,1,1],[4,-2,5],[2,8,7]])
print(b)
print(np.linalg.det(b))

#numpy.linalg.solve()函数给出了矩阵形式的线性方程的解。
#AX=B
#X=A^(-1)*B

#numpy.linalg.inv()函数来计算矩阵的逆。
# 矩阵的逆是这样的，如果它乘以原始矩阵，则得到单位矩阵
a=np.array([[1,2],[3,4]])
y=np.linalg.inv(a)
print(y)
print(a)
print(np.dot(a,y))

a=np.array([[1,1,1],[0,2,5],[2,5,-1]])
print(a)
ainv=np.linalg.inv(a)
#a的逆
print(ainv)
b=np.array([[6],[-4],[27]])
print(b)
#计算X=A^(-1)*B
x=np.linalg.solve(a,b)
print(x)

#numpy.histogram()函数将输入数组和bin作为两个参数。 bin数组中的连续元素用作每个bin的边界。
# a = np.array([22,87,5,43,56,73,55,54,11,20,51,5,79,31,27])
# hist,bins = np.histogram(a,bins =  [0,20,40,60,80,100])  
# print(hist)
# #由输出可见，改函数自动提取了数组a中符合bins每个区间数据个数
# print(bins)
# #这里可以看到np.histogram可以分别把它的两个参数值赋值给两个参数
# plt.hist(a,bins)
# #这里通过plt可视化可以看到hitsogram中bins参数的作用
# plt.show()

#由输出可见，改函数自动提取了数组a中符合bins每个区间数据个数
# load()和save()函数处理 numPy 二进制文件（带npy扩展名）
# loadtxt()和savetxt()函数处理正常的文本文件

#numpy.save()文件将输入数组存储在具有npy扩展名的磁盘文件中。
# a = np.array([1,2,3,4,5]) 
# np.save('outfile',a)
# #为了从outfile.npy重建数组，请使用load()函数。
# b = np.load('outfile.npy') 
# print(b)

# #保存为文本文件
# np.savetxt('out.txt',a) 
# b = np.loadtxt('out.txt') 
# print(b) 