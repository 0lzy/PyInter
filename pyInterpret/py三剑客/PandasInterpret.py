import pandas as pd
import numpy as np
import numpy.matlib

# #Series 结构，也称 Series 序列，是 Pandas 常用的数据结构之一，
# #它是一种类似于一维数组的结构，由一组数据值（value）和一组标签组成，其中标签与数据值之间是一一对应的关系。
# #Series 可以保存任何数据类型，比如整数、字符串、浮点数、Python 对象等，
# #它的标签默认为整数，从 0 开始依次递增。

# #Pandas 使用 Series() 函数来创建 Series 对象，通过这个对象可以调用相应的方法和属性，从而达到处理数据的目的：
# # data	输入的数据，可以是列表、常量、ndarray 数组等。
# # index	索引值必须是惟一的，如果没有传递索引，则默认为 np.arrange(n)。
# # dtype	dtype表示数据类型，如果没有提供，则会自动判断得出。
# # copy	表示对 data 进行拷贝，默认为 False。#输出数据为空

# #创建一个空的Series对象
# s = pd.Series()
# print(s)

# # ndarray创建Series对象（一维数据结构）
# #ndarray 是 NumPy 中的数组类型，
# # 当 data 是 ndarry 时，传递的索引必须具有与数组相同的长度。假如没有给 index 参数传参，
# # 在默认情况下，索引值将使用是 range(n) 生成，其中 n 代表数组长度，如下所示：
# #n=np.arange(len(arr)-1)
# a=np.arange(5,dtype=np.float16)
# s=pd.Series(a)
# print(s)
# #结构图如下：
# #标签      数据值
# # .          .
# # .          .
# # .          .
# #  数据值类型

# #写入为dict类型时，在没有指定索引的情况下，会把前键当成索引值
# data = {'a' : 0., 'b' : 1., 'c' : 2.}
# s = pd.Series(data)
# print(s)
# s=pd.Series(data,index=['b','c','h','a'])
# print(s)
# #索引值唯一，以提供的索引值顺序index优先,
# #当传递的索引值无法找到与其对应的值时，使用 NaN（非数字）填充。

# b=np.array([2,3,4,5,6])
# s=pd.Series(a,index=b)
# #这里b充当索引，必须和数组长度相同，否则会报错
# #print(s[0])
# #以上这个查找方法会报错
# #当所给标签与默认标签数据类型相同时，只可以用标签索引
# print(s[5])
# #这里可以给出一个char类型的数组标签
# c=['a','b','c','d','e']
# s=pd.Series(a,index=c)
# print(s[0])   #位置索引
# print(s['a'])  #标签索引
# #这里可以看到两种查找方式都可以使用了

# #标量创建Series对象
# #如果data是标量值，则必须提供索引
# s=pd.Series(5,index=[0,1,2,3])
# print(s)
# #标量值按照index的数量进行重复，并与其一一对应

# #访问Series数据
# a=np.array([1,2,3,4,5])
# s=pd.Series(a,index=['a','b','c','d','e'],dtype=np.int16)
# print(s[:3])
# #左闭右开，返回前三个元素值

# #获取最后三个元素
# print(s[-3:])

# #索引标签访问
# a=np.array([1,2,3,4,5])
# s=pd.Series(a,index=['a','b','c','d','e'])
# #访问单个
# print(s['a'])
# #访问多个
# #多个索引标签需要以数组形式输入
# print(s[['a','b','c']])

# #如果index中不包含标签，则会触发异常
# #print(s['f'])

# #下面我们介绍 Series 的常用属性和方法。在下表列出了 Series 对象的常用属性。
# # axes	以列表的形式返回所有行索引标签。
# # dtype	返回对象的数据类型。
# # empty	返回一个空的 Series 对象。
# # ndim	返回输入数据的维数。
# # size	返回输入数据的元素数量。
# # values	以 ndarray 的形式返回 Series 对象。
# # index	返回一个RangeIndex对象，用来描述索引的取值范围。
# s=pd.Series(np.random.randn(5))
# print(s)
# print(s.axes)
# #返回索引标签起始点和截止点（不包含截止点），和步进值
# print(s.index)
# #index与axes作用类似，不同的是axes会返回一个数组类型
# print(s.dtype)
# #返回数据类型
# print(s.empty)
# #empty返回一个布尔值，用于判断数据对象是否为空
# #ndim查看序列的维数。根据定义，Series 是一维数据结构，因此它始终返回 1。
# print(s.ndim)
# #返回Series对象的大小（长度）
# print(s.size)
# #以一维数组形式返回Series对象（ndarray）
# print(s.values)

# #查看数据
# # 如果想要查看 Series 的某一部分数据，可以使用 head() 或者 tail() 方法。
# # 其中 head() 返回前 n 行数据，默认显示前 5 行数据。
# s=pd.Series(np.random.randn(5))
# print(s.head())
# print(s.head(3))
# #仍然以Series结构输出

# #tail() 返回的是后 n 行数据，默认为后 5 行
# #注意虽然是返回最后n行，但仍以Series顺序输出
# print(s.tail())
# print(s.tail(3))

# #检测缺失值
# #isnull()：如果为值不存在或者缺失，则返回 True。
# #notnull()：如果值不存在或者缺失，则返回 False。
# #仍然以Series结构输出
# s=pd.Series([1,2,5,None])
# #这里我们人工定义一个缺失值None
# print(pd.isnull(s))
# print(pd.notnull(s))


# # DataFrame 一个表格型的数据结构，既有行标签（index），又有列标签（columns），
# # 它也被称异构数据表，所谓异构，指的是表格中每列的数据类型可以不同，比如可以是字符串、整型或者浮点型等。
# #DataFrame 的每一行数据都可以看成一个 Series 结构，只不过，DataFrame 为这些行中每个数据值增加了一个列标签。
# #同 Series 一样，DataFrame 自带行标签索引，默认为“隐式索引”即从 0 开始依次递增，行标签与 DataFrame 中的数据项一一对应。
# # DataFrame 每一列的标签值允许使用不同的数据类型；
# # DataFrame 是表格型的数据结构，具有行和列；
# # DataFrame 中的每个数据值都可以被修改。
# # DataFrame 结构的行数、列数允许增加或者删除；
# # DataFrame 有两个方向的标签轴，分别是行标签和列标签；
# # DataFrame 可以对行和列执行算术运算。

# #pd.DataFrame( data, index, columns, dtype, copy)
# # data	输入的数据，可以是 ndarray，series，list，dict，标量以及一个 DataFrame。
# # index	行标签，如果没有传递 index 值，则默认行标签是 np.arange(n)，n 代表 data 的元素个数。
# # columns	列标签，如果没有传递 columns 值，则默认列标签是 np.arange(n)。
# # dtype	dtype表示每一列的数据类型。
# # copy	默认为 False，表示复制数据 data。

# #创建DataFame对象
# #使用下列方式创建一个空的 DataFrame，这是 DataFrame 最基本的创建方法。
# df = pd.DataFrame()
# print(df)

# #可以使用单一列表或嵌套列表来创建一个 DataFrame。
# data=[1,2,3,4,5]
# df=pd.DataFrame(data)
# print(df)
# #这里默认提供了行标签，没有传入列标签

# #使用嵌套列表创建 DataFrame 对象
# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'])
# #列标签数量应该与数据最小轴数据量相同
# print(df)

# #指定数值元素的数据类型
# #使用数组类型创建
# data = [['Alex',10],['Bob',12],['Clarke',13]]
# df = pd.DataFrame(data,columns=['Name','Age'],dtype=np.float16)
# print(df)

# #字典嵌套列表创建
# data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
# # data 字典中，键对应的值的元素长度必须相同（也就是列表长度相同）
# #如果不相同会报错，如下,前四后三：
# #data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29]}
# # 如果传递了索引，那么索引的长度应该等于数组的长度；如果没有传递索引，那么默认情况下，索引将是 range(n)，其中 n 代表数组长度。
# df=pd.DataFrame(data)
# print(df)
# #给上例添加行标签index
# data = {'Name':['Tom', 'Jack', 'Steve', 'Ricky'],'Age':[28,34,29,42]}
# df = pd.DataFrame(data, index=['rank1','rank2','rank3','rank4'])
# print(df)

# #列表嵌套字典创建DataFrame对象
# data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
# df=pd.DataFrame(data=data,index=['first', 'second'])
# print(df)

# data = [{'a': 1, 'b': 2},{'a': 5, 'b': 10, 'c': 20}]
# df1 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b'])
# df2 = pd.DataFrame(data, index=['first', 'second'], columns=['a', 'b1'])
# print(df1)
# print(df2)
# #列标签没有对应的数据值，返回NaN

# # Series创建DataFrame对象
# # 您也可以传递一个字典形式的 Series，从而创建一个 DataFrame 对象，其输出结果的行索引是所有 index 的合集。
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# print(a)
# print(b)
# print(d)
# df=pd.DataFrame(d)
# print(df)
# #对于 one 列而言，此处虽然显示了行索引 ‘d’，但由于没有与其对应的值，所以它的值为 NaN。

# #列索引操作DataFrame
# #DataFrame 可以使用列索（columns index）引来完成数据的选取、添加和删除操作。
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# df=pd.DataFrame(d)
# print(df)
# #列索引以Series结构输出
# print(df['one'])
# print(df['two'])

# #使用 columns 列索引表标签可以实现添加新的数据列，
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# df=pd.DataFrame(d)
# print(df)
# df['three']=pd.Series([10,20,30],index=['a','b','c'])
# #甚至可以相加
# df['four']=df['one']+df['three']
# print(df)

# #上述示例，我们初次使用了 DataFrame 的算术运算，这和 NumPy 非常相似。
# # 除了使用df[]=value的方式外，您还可以使用 insert() 方法插入新的列，
# data=[['Jack',18],['Helen',19],['John',17]]
# df=pd.DataFrame(data,columns=['name','age'])
# print(df)
# #注意是column参数
# #数值1代表插入到columns列表的索引位置
# df.insert(1,column='score',value=[91,90,75])
# print(df)

# #列索引删除数据列
# #通过 del 和 pop() 都能够删除 DataFrame 中的数据列。
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# df=pd.DataFrame(d)
# print(df)
# del(df['one'])
# print(df)
# df.pop('two')
# print(df)
# #这里删没了，变成了空DataFrame数据结构

# #行索引操作DataFrame
# #可以将行标签传递给 loc 函数，来选取数据。
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# df=pd.DataFrame(d)
# print(df)
# #这里仍然以Series结构输出
# print(df.loc['a']) 
# #不能想获取列标签那样直接接受索引，如下：
# #print(df['a'])

# #loc 允许接两个参数分别是行和列，参数之间需要使用“逗号”隔开，但该函数只能接收标签索引。
# print(df.loc['b','one'])
# #这里返回单个数据对象

# #通过将数据行所在的索引位置传递给 iloc 函数，也可以实现数据行选取
# print(df.iloc[2])
# print(df.iloc[1])
# #以上返回每行的第整数个元素位置

# #切片操作多行选取
# #左闭右开
# a=pd.Series([1, 2, 3], index=['a', 'b', 'c'])
# b=pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
# d = {'one' : a,'two':b}
# df=pd.DataFrame(d)
# print(df[2:4])
# #print(df[2])这种不行

# #使用 append() 函数，可以将新的数据行添加到 DataFrame 中，该函数会在行末追加数据行。
# df= pd.DataFrame([[1, 2], [3, 4]], columns = ['a','b'])
# df2 = pd.DataFrame([[5, 6], [7, 8]], columns = ['a','b'])
# print(df)
# print(df2)
# #将df2堆叠或者添加到df1的尾部(追加数据行)
# #行标签不会刷新，而是按照df2原来的行标签进行堆叠
# print(df.append(df2))
# #我们在这里重新查看一下原DataFrame结构，发现没有改变
# print(df)

# #您可以使用行索引标签，从 DataFrame 中删除某一行数据。如果索引标签存在重复，那么它们将被一起删除。
# #例如上例堆叠后出现重复标签
# df=df.drop(0)
# print(df)
# #由于原DataFrame结构没有改变，所以我们在这里只删除了一个0行，输出了一个1行
# print(df.append(df2).drop(0))
# #这句才是删除了堆叠之后的0行，剩余两个1行

# #DataFrame 的属性和方法，与 Series 相差无几，如下所示：
# # T	行和列转置。
# # axes	返回一个仅以行轴标签和列轴标签为成员的列表。
# # dtypes	返回每列数据的数据类型。
# # empty	DataFrame中没有数据或者任意坐标轴的长度为0，则返回True。
# # ndim	轴的数量，也指数组的维数。
# # shape	返回一个元组，表示了 DataFrame 维度。
# # size	DataFrame中的元素数量。
# # values	使用 numpy 数组表示 DataFrame 中的元素值。
# # head()	返回前 n 行数据。
# # tail()	返回后 n 行数据。
# # shift()	将行或列移动指定的步幅长度

# #创建一个DataFrame对象
# d = {'Name':pd.Series(['c语言中文网','编程帮',"百度",'360搜索','谷歌','微学苑','Bing搜索']),
#    'years':pd.Series([5,6,15,28,3,19,23]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8])}
# df=pd.DataFrame(d)
# print(df)
# #以上默认生成了行标签，ndarray类型，np.arangr(7)

# #T（Transpose）转置
# print(df.T)
# #在这里标签也会跟着数据值一起转置，所以原来的数据结构没有发生变化
# #根据下面的dtypes方法得知，T转置这个方法没有改变原来的数据结构，
# #换句话说，df还是原来的df

# #axes返回一个行标签、列标签组成的列表。
# #如果行标签为默认的ndarray类型，则会在列表返回时给出起始值，结束值，步进值（arange)
# print(df.axes)

# #dtypes返回每一列的数据类型。
# print(df.dtypes) 

# # empty返回一个布尔值，判断输出的数据对象是否为空，若为 True 表示对象为空
# print(df.empty) #这里显然数据对象非空

# #ndim返回数据对象的维数。
# print(df.ndim)
# #DataFrame数据结构默认是2维的，这里永远输出的是2
# #如果是Series结构的话，这里的返回值是1


# #shape返回一个代表 DataFrame 维度的元组。返回值元组 (a,b)，其中 a 表示行数，b 表示列数。
# print(df.shape)
# #这里的shape和numpy中的作用类似，都是返回一个数组维度的元组类型

# #size返回 DataFrame 中的元素数量
# print(df.size)
# #可以用上述shape的两个返回值的乘积来计算
# #如下;
# print(df.shape[0]*df.shape[1])

# #values以 ndarray 数组的形式返回 DataFrame 中的数据。
# print(df.values)
# #这里返回的是二维数组类型

# #head()&tail()查看数据
# #与Series中的方法相似
# #注意这里只会遍历行
# #默认值为5
# print(df.head(3))

# #shift()移动行或列
# # 如果您想要移动 DataFrame 中的某一行/列，可以使用 shift() 函数实现。
# # 它提供了一个periods参数，该参数表示在特定的轴上移动指定的步幅。

# # peroids	类型为int，表示移动的幅度，可以是正数，也可以是负数，默认值为1。
# # freq	日期偏移量，默认值为None，适用于时间序。取值为符合时间规则的字符串。
# # axis	如果是 0 或者 “index” 表示上下移动，如果是 1 或者 “columns” 则会左右移动。
# # fill_value	该参数用来填充缺失值。
# #DataFrame.shift(periods=1, freq=None, axis=0)
# info= pd.DataFrame({'a_data': [40, 28, 39, 32, 18], 
# 'b_data': [20, 37, 41, 35, 45], 
# 'c_data': [22, 17, 11, 25, 15]}) 
# print(info)
# print(info.shift(periods=1,axis=0))
# #由这一次可知period默认向下移动
# print(info.shift(periods=3))
# #没有指定轴时，默认为0
# #将缺失值和原数值替换为52
# print(info)
# print(info.shift(periods=3,axis=1))
# print(info.shift(periods=3,axis=1,fill_value= 52))  
# print(info.shift(periods=3).shift(periods=1,axis=1))
# print(info.shift(periods=3,fill_value=1).shift(periods=1,axis=1,fill_value=0))
# #以上这个例子返回了原来的DataFrame数据先向下移动3，再向左移动1，分别填充缺失值，
# #以上这个例子中，第二次填充的缺失值替换掉了第一次填充的缺失值
# #说明fill_value 参数不仅可以填充缺失值，还也可以对原数据进行替换。
# print(info)
# print(info.shift(fill_value=1))
# for i in range(5):
#     df=info
#     info=df.shift(fill_value=1)
# print(info)
# #由以上可知，当shift方法中只有添加确实值时，其他参数都为默认，即period=1，axis=0
# #循环5次之后，数据对象全部被替换

# #自 Pandas 0.25 版本后， Panel 结构已经被废弃
# #Panel 是一个用来承载数据的三维数据结构，它有三个轴，分别是 items（0 轴），major_axis（1 轴），而 minor_axis（2 轴）。
# # items：axis =0，Panel 中的每个 items 都对应一个 DataFrame。
# # major_axis：axis=1，用来描述每个 DataFrame 的行索引。
# # minor_axis：axis=2，用来描述每个 DataFrame 的列索引。
# #pandas.Panel(data, items, major_axis, minor_axis, dtype, copy

# # #创建一个空Panel
# # p=pd.Panel()
# # print(p)

# # #返回均匀分布的随机样本值位于[0,1）之间
# # data = np.random.rand(2,4,5)
# # print(data)
# # p = pd.Panel(data)
# # print(p)
# # #相比于空的Panel，上例这里返回了传入的数组的三个轴的维数，

# # #下面使用 DataFrame 创建一个 Panel
# # data = {'Item1' : pd.DataFrame(np.random.randn(4, 3)),
# #    'Item2' : pd.DataFrame(np.random.randn(4, 2))}
# # p = pd.Panel(data)
# # print(p)

# # #Panel中选取数据
# # #如果想要从 Panel 对象中选取数据，可以使用 Panel 的三个轴来实现，也就是items，major_axis，minor_axis。
# # #使用items选取数据
# # data = {'Item1':pd.DataFrame(np.random.randn(4, 3)),
# #    'Item2':pd.DataFrame(np.random.randn(4, 2))}
# # p = pd.Panel(data)
# # #以DataFrame二位数据结构输出
# # print(p['Item1'])


# #Python Pandas描述性统计
# # count()	统计某个非空值的数量。
# # sum()	求和
# # mean()	求均值
# # median()	求中位数
# # mode()	求众数
# # std()	求标准差
# # min()	求最小值
# # max()	求最大值
# # abs()	求绝对值
# # prod()	求所有数值的乘积。
# # cumsum()	计算累计和，axis=0，按照行累加；axis=1，按照列累加。
# # cumprod()	计算累计积，axis=0，按照行累积；axis=1，按照列累积。
# # corr()	计算数列或变量之间的相关系数，取值-1到1，值越大表示关联性越强。

# #sum()求和,在默认情况下，返回 axis=0 的所有值的和。
# #创建字典型series结构
# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# print(df)
# print(df.sum())
# #以上这个会以Series结构输出
# # sum() 和 cumsum() 函数可以同时处理数字和字符串数据。
# # 虽然字符聚合通常不被使用，但使用这两个函数并不会抛出异常；
# # 而对于 abs()、cumprod() 函数则会抛出异常，因为它们无法操作字符串数据。

# #下面再看一下 axis=1 的情况
# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# #也可使用sum("columns")或sum(1)
# print(df.sum(axis=1))
# #这里相加的时候直接舍弃了第一列的字符串

# #mean()求均值
# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,30,29,23,34,40,30,51,46]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# print(df.mean())
# #由输出结果可知这里默认轴为axis=0

# #std()求标准差返回数值列的标准差
# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,59,19,23,44,40,30,51,54]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# print(df.std())

# #数据汇总描述
# #describe() 函数显示与 DataFrame 数据列相关的统计信息摘要。
# #求出数据的所有描述信息
# print(df.describe())
# #describe() 函数输出了平均值、std 和 IQR 值(四分位距)等一系列统计信息。

# #通过 describe() 提供的include能够筛选字符列或者数字列的摘要信息。
# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,59,19,23,44,40,30,51,54]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# # object： 表示对字符列进行统计信息描述；
# print(df.describe(include=["object"]))

# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,59,19,23,44,40,30,51,54]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# # number：表示对数字列进行统计信息描述；
# df.plot()
# print(df.describe(include=["number"]))

# d = {'Name':pd.Series(['小明','小亮','小红','小华','老赵','小曹','小陈',
#    '老李','老王','小冯','小何','老张']),
#    'Age':pd.Series([25,26,25,23,59,19,23,44,40,30,51,54]),
#    'Rating':pd.Series([4.23,3.24,3.98,2.56,3.20,4.6,3.8,3.78,2.98,4.80,4.10,3.65])
# }
# df = pd.DataFrame(d)
# # all：汇总所有列的统计信息
# print(df.describe(include="all"))


# #Python Pandas绘图
# #Pandas 对 Matplotlib 绘图软件包的基础上单独封装了一个plot()接口，通过调用该接口可以实现常用的绘图操作。
# a=np.random.rand(8,4)
# print(a)
# #创建包含时间序列的数据
# df=pd.DataFrame(a,index=pd.date_range('2/1/2020',periods=8),columns=list('ABCD'))
# print(df)
# df.plot()
# # 如上图所示，如果行索引中包含日期，Pandas 会自动调用 gct().autofmt_xdate() 来格式化 x 轴。
# #此方法默认使用线条绘制折线图

# # 柱状图：bar() 或 barh()
# # 直方图：hist()
# # 箱状箱：box()
# # 区域图：area()
# # 散点图：scatter()
# #通过关键字参数kind可以把上述方法传递给 plot()。

# #创建一个柱状图
# df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])\
# #这里的列标签个数一定要和随机生成的数组的第二维度元素个数相同,否则会报错
# print(df)
# df.plot(kind="barh")
# df.plot.bar()
# #两种方式都可以输出图形,bar为纵向柱状图,barh为横向柱状图
# #这里的colunms列标签就是绘图中的图例
# #df.plot.bar(stacked=True)
# #或者
# df.plot(kind="bar",stacked=True)
# #这里通过设置参数stacked=True可以生成柱状堆叠图
# df.plot(kind='barh',stacked=True)

# #直方图
# df = pd.DataFrame({'A':np.random
#                    .randn(100)+2,'B':np.random.randn(100),'C':
# np.random.randn(100)-2}, columns=['A', 'B', 'C'])
# #注意以上用于生成数据的函数为np.random.randn
# #np.random.randn()用于生成指定维度的、从标准正态分布中抽取出来的随机数（均值为0，标准差为1）。
# #该函数的参数是表示维度的整数或整数序列，返回一个对应维度且值为标准正态分布中抽取出来的随机数的数组。
# #如上述例子：其中第一列的值是从一个均值为2，标准差为1的正态分布中生成出来的，
# # 第二列的值则是从均值为0，标准差为1的正态分布中生成出来的。
# # 而不是np.random.rand（用于生成一个具有随机值得数组或矩阵）
# print(df)
# #指定箱数为15
# #将数据分成15个区间，显示它们的频率分布情况
# df.plot.hist(bins=15)
# # 在直方图中，箱数指的是将数据分成几个区间来绘制其频率分布。
# # 每个区间内的数据称为一个“箱子”。箱数越多，即箱子数目越多，则表示对数据的刻画更加详细，
# # 但同时也可能会使得图形过于拥挤，难以看清楚各个箱子之间的差异。
# # 在这里使用参数bins=15时，将总的数据范围分成了15个小区间（即箱子数），并计算出每个小区间的频数或频率。
# # 因此，每个直方图柱状体所展示的就是该数据范围内每一个小区间的频数或频率，
# # 它们高度不尽相同，因为它们代表的数据取值范围不同。
# # 在可视化分析中，适当地选择箱数可以有效地呈现数据的概貌和结构特征，有利于识别异常数据点和离群值。

# df = pd.DataFrame({'A':np.random.randn(100)+2,
#                    'B':np.random.randn(100),
#                    'C':np.random.randn(100)-2,
#                    'D':np.random.randn(100)+3},
#                    columns=['A', 'B', 'C','D'])
# #列标签用于生成图例
# df.diff().hist(color="r",alpha=0.5,bins=15)
# #alpha表示透明度

# df = pd.DataFrame(np.random.rand(10,4),columns=['a','b','c','d'])
# df.plot(kind='box')
# #这里随机生成4组数据，每组10个数据，箱型表示该数据范围内每一个数据的频数或频率（区间）

# #区域图
# df = pd.DataFrame(np.random.rand(5,4),columns=['a','b','c','d'])
# df.plot(kind='area')

# #散点图
# #30个数据
# df = pd.DataFrame(np.random.rand(30, 4), columns=['a', 'b', 'c', 'd'])
# #df.plot.scatter(x='a',y='b')
# df.plot(kind='scatter',x='a',y='b')
# #这里传入x轴和y轴的名称
# #参数"x"和"y"必须指定DataFrame中存在的列名
# #df.plot(kind='scatter',x='x',y='y')
# #以上这个就会报错

#饼状图
df = pd.DataFrame(np.random.rand(4),index=['go','java','c++','c'],columns=['L'])
#这里列标签用于命名子图
print(df)   
df.plot.pie(subplots=True)
#参数"subplots=True"表示将每个数据列单独绘制成一个子图，每个子图都是一个独立的、完整的饼图。

#Pandas csv读写文件
df=pd.read_csv('D:/PYTHONPROJECTS/tips.csv')
print(df)
# 自定义索引
#在 CSV 文件中指定了一个列，然后使用index_col可以实现自定义索引。
df=pd.read_csv("D:/PYTHONPROJECTS/tips.csv",index_col=['tip'])
# df=pd.read_csv('D:/PYTHONPROJECTS/tips.csv')
# df=df.set_index('tip')
#这两行与上作用相同
print(df)


df=pd.read_csv('D:/PYTHONPROJECTS/tips.csv')
print(df)
print(df.dtypes)
#设置某一列的数据了类型
df=pd.read_csv('D:/PYTHONPROJECTS/tips.csv',dtype={'size':np.float64})
print(df)
print(df.dtypes)
#可以看到size列数据类型已经变成float64

#更改文件头名
#使用names参数可以指定头文件名
df=pd.read_csv("D:/PYTHONPROJECTS/tips.csv",names=['A','B','C','D'])
print(df)
#这里被标记的列标签数小于文件原有的列标签数，只标记了后四个，同时还保留了原来的列标签，但是没有了行标签
#如下，所有的头文件都被指定后，才保留了DataFrame结构
df=pd.read_csv("D:/PYTHONPROJECTS/tips.csv",names=['A','B','C','D','E','F','G'])
print(df)
#文件标头名是附加的自定义名称，但是您会发现，原来的标头名（列标签名）并没有被删除，
# 此时您可以使用header参数来删除它。
df=pd.read_csv("D:/PYTHONPROJECTS/tips.csv",names=['A','B','C','D','E','F','G'],header=0)
print(df)

#跳过指定行数
#skiprows
df=pd.read_csv("D:/PYTHONPROJECTS/tips.csv",names=['A','B','C','D','E','F','G'],header=0,skiprows=243)
print(df)
#跳过后，行标签从跳过之后的第一行开始计算，即位0
#注意包含标头所在行

#to_csv
#Pandas 提供的 to_csv() 函数用于将 DataFrame 转换为 CSV 数据。
# 如果想要把 CSV 数据写入文件，只需向函数传递一个文件对象即可。否则，CSV 数据将以字符串格式返回。
data={'name':['Smith','Parker'],'ID':[101,102],'language':['Python','JavaScript']}
df=pd.DataFrame(data)
print(df)
cvs_data=df.to_csv()
print(cvs_data)
#np.savetxt('a.txt',df)
#我怀疑这个只能保存array,在这里用会报错
#转换为csv数据

#指定 CSV 文件输出时的分隔符，并将其保存在 pandas.csv 文件中，代码如下：
csv_file=df.to_csv('D:/PYTHONPROJECTS/pandas.csv',sep='|')
#如果使用了分隔符，这个函数不会自动识别excel的分区
#如果不使用分隔符，就会按照excel的格子自动分区输出


#andas Excel读写操作详解
#通过 to_excel() 函数可以将 Dataframe 中的数据写入到 Excel 文件。
#如果想要把单个对象写入 Excel 文件，那么必须指定目标文件名；
# 如果想要写入到多张工作表中，则需要创建一个带有目标文件名的ExcelWriter对象，并通过sheet_name参数依次指定工作表的名称。
#DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', float_format=None, columns=None, header=True, index=True, index_label=None, startrow=0, startcol=0, engine=None, merge_cells=True, encoding=None, inf_rep='inf', verbose=True, freeze_panes=None)  

#excel_wirter	文件路径或者 ExcelWrite 对象。
# sheet_name	指定要写入数据的工作表名称。
# na_rep	缺失值的表示形式。
# float_format	它是一个可选参数，用于格式化浮点数字符串。
# columns	指要写入的列。
# header	写出每一列的名称，如果给出的是字符串列表，则表示列的别名。
# index	表示要写入的索引。
# index_label	引用索引列的列标签。如果未指定，并且 hearder 和 index 均为为 True，则使用索引名称。如果 DataFrame 使用 MultiIndex，则需要给出一个序列。
# startrow	初始写入的行位置，默认值0。表示引用左上角的行单元格来储存 DataFrame。
# startcol	初始写入的列位置，默认值0。表示引用左上角的列单元格来储存 DataFrame。
# engine	它是一个可选参数，用于指定要使用的引擎，可以是 openpyxl 或 xlsxwriter。

#创建DataFrame数据
df = pd.DataFrame({'name': ['编程帮', 'c语言中文网', '微学苑', '92python'],
     'rank': [1, 2, 3, 4],
     'language': ['PHP', 'C', 'PHP','Python' ],
     'url': ['www.bianchneg.com', 'c.bianchneg.net', 'www.weixueyuan.com','www.92python.com' ]})
pd.set_option('display.unicode.ambiguous_as_wide', True)  #处理数据的列标题与数据无法对齐的常规情况
pd.set_option('display.unicode.east_asian_width', True)   #无法对齐主要是因为列标题是中文时使用这个
print(df)
#创建ExcelWrite对象
# writer = pd.ExcelWriter('website.xlsx')
# df.to_excel(writer)
#write.save()
df.to_excel('D:/PYTHONPROJECTS/website1.xlsx')
#可以看到仅仅用上面这一句也可以实现同样的效果
# print('输出成功')


#read_excel()如果您想读取 Excel 表格中的数据，可以使用 read_excel() 方法，其语法格式如下：

# pd.read_excel(io, sheet_name=0, header=0, names=None, index_col=None,
#               usecols=None, squeeze=False,dtype=None, engine=None,
#               converters=None, true_values=None, false_values=None,
#               skiprows=None, nrows=None, na_values=None, parse_dates=False,
#               date_parser=None, thousands=None, comment=None, skipfooter=0,
#               convert_float=True, **kwds)


# io	表示 Excel 文件的存储路径。
# sheet_name	要读取的工作表名称。
# header	指定作为列名的行，默认0，即取第一行的值为列名；若数据不包含列名，则设定 header = None。若将其设置 为 header=2，则表示将前两行作为多重索引。
# names	一般适用于Excel缺少列名，或者需要重新定义列名的情况；names的长度必须等于Excel表格列的长度，否则会报错。
# index_col	用做行索引的列，可以是工作表的列名称，如 index_col = ‘列名’，也可以是整数或者列表。
# usecols	int或list类型，默认为None，表示需要读取所有列。
# squeeze	boolean，默认为False，如果解析的数据只包含一列，则返回一个Series。
# converters	规定每一列的数据类型。
# skiprows	接受一个列表，表示跳过指定行数的数据，从头部第一行开始。
# nrows	需要读取的行数。
# skipfooter	接受一个列表，省略指定行数的数据，从尾部最后一行开始。


#读取excel数据
df=pd.read_excel('website.xlsx')
print(df)
df = pd.read_excel('website.xlsx',index_col='name',skiprows=[2])
#这里skiprows接受一个列表，只跳过了指定的第二行
print(df)
# #处理未命名列
df.columns = df.columns.str.replace('Unnamed.*', 'col_label')
print(df)

df = pd.read_excel('website.xlsx')
print(df)
#读取excel数据
#index_col选择前两列作为索引列
#选择前三列数据，name列作为行索引
df = pd.read_excel('website.xlsx',index_col=[0,1],usecols=[1,2,3])
#这里选取前三列中不过包括 未命名列
#处理未命名列，固定用法
df.columns = df.columns.str.replace('Unnamed.*', 'col_label')
print(df)


#转换ndarray数组
# 在某些情况下，需要执行一些 NumPy 数值计算的高级函数，这个时候您可以使用 to_numpy() 函数
# 将 DataFrame 对象转换为 NumPy ndarray 数组，并将其返回。函数的语法格式如下：
# DataFrame.to_numpy(dtype=None, copy=False)
# dtype：可选参数，表示数据类型；
# copy：布尔值参数，默认值为 Fales，表示返回值不是其他数组的视图。
df = pd.DataFrame({"P": [2, 3], "Q": [4.0, 5.8]})
print(df)
#给info添加R列 
df['R'] = pd.date_range('2020-12-23', periods=2)
#生成时间序列，长度为2
print(df)
n=df.to_numpy()
print(n)
#输出可以看到已经转换为ndarray类型吗，还是个二维的，对应DataFram结构
print(type(n))

#创建DataFrame对象
info = pd.DataFrame([[17, 62, 35],[25, 36, 54],[42, 20, 15],[48, 62, 76]], columns=['x', 'y', 'z']) 
print(info)
n=info.to_numpy()
print(n)