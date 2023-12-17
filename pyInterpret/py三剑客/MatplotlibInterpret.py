import numpy as np
import numpy.matlib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import mpl_toolkits.axisartist as axisartist
import math

#figure
#在绘图之前，我们需要一个figure对象，可以理解成我们需要一张画布才能开始绘图
#fig=plt.figure()  #设置一个画布

#Axes
#拥有figure后，我们还需要创建绘图区域，添加Axes
#在绘制子图的过程中，对于每一个字图肯能有不同的设置
#Axes可以直接实现对于单个字图的设定

#添加一个Axes，参数111可以理解为在第一行第一列的第一个位置生成一个Axes对象来准备作画
#也可以是fig.add_subplot(2,2,1)的方式生成Axes，前面两个参数确定了面板的划分
#以上这个方法中可以有逗号，可以没有
#例如2，2会将整个画板划分成2*2的方格，第三个参数的取值范围是[1，2*2]，表示第几个Axes
# fig=plt.figure()
# ax1=fig.add_subplot(221)
# ax2=fig.add_subplot(222)
# ax3=fig.add_subplot(224)
# ax3.set(xlim=[0.5,4.5],ylim=[-2,8],title="An Example Axes",ylabel='Y-axis',xlabel='x-axis')
# plt.show()

# fig,axes=plt.subplots(nrows=2,ncols=2)
# #表示生成的画布被分成2行2列
# axes[0,0].set(xlim=[0.5,4.5],ylim=[-2,8],title="An Example Axes",ylabel='Y-axis',xlabel='x-axis')
# #这个方法中仍然以传入坐标间距，坐标名称，画布名称这些参数
# axes[0,1].set(title='2')
# axes[1,0].set(title='3')
# axes[1,1].set(title='4')
# plt.show()
#fig还是我们熟悉的面板，axes成了我们常用二维数组的形式访问

#pyplot
#下面采用pyplot绘图，但只适合简单的绘图，可以快速地将草图绘出
#处理复杂的绘图工作，特别是有多个字图时，最好还是用Axes完成绘画
# a=np.array([1,2,3,4])
# b=np.array([10,20,25,30])
# plt.plot(a,b,color='lightblue',linewidth=3)
# #简单的草图不能设置精确的坐标参数,但可以设置图像颜色，宽度等
# #传入的数据可以是ndarray数组，也可以是列表例如：
# #传入的数据范围即是坐标轴的范围
# c=[1,2,3,4]
# d=[10,20,25,30]
# plt.plot(c,d,color='red',linewidth=3)
# plt.show()


#设置画布大小
#在使用matplotlib作图时，会遇到图片显示不全或者图片大小不是我们想要的，这个时候就需要调整画布大小。
#下例左图为500*500像素，右图为1000*1000像素。
# 500 x 500 像素（先宽度 后高度）
# 注意这里的宽度和高度的单位是英寸，1英寸=100像素
# fig=plt.figure(figsize=(5,5))
# ax=fig.add_subplot(111)
# plt.show()


#设置网格线
#通过 axes 对象提供的 grid() 方法可以开启或者关闭画布中的网格以及网格的主/次刻度。
#除此之外，grid() 函数还可以设置网格的颜色、线型以及线宽等属性。
#grid(color='b', ls = '-.', lw = 0.25)
# ls：表示网格线的样式；
# lw：表示网格线的宽度；
#fig画布，axes子图区域
# fig,axes=plt.subplots(1,3,figsize=(12,4))
# x=np.arange(1,11)
# axes[0].plot(x,x**3,'g',lw=2)
# #以上在这里的lw就代表所绘制图像线条宽度，g为绿色
# #开启网格
# axes[0].grid(True)
# #axes[0].set(title='default grid')
# axes[0].set_title('default grid')
# #以上两种设置图像标题的方法
# axes[1].plot(x, np.exp(x), 'r')
# # 设置网格的颜色，线型，线宽
# axes[1].grid(color='b', ls='-.', lw=0.25)
# axes[1].set_title('custom grid')
# axes[2].plot(x,x)
# axes[2].set_title('no grid')
# fig.tight_layout()  #自动调节图像位置，使其美观
# plt.show()

#设置坐标轴
#set_xlabel 用字符串列表来设置坐标轴的标签，
# fontdict 设置轴标签的字体和字号等参数。
# fontdict = {'weight': 'normal', 'family': 'Times New Roman', 'size': 20}

# # fig,axes=plt.subplots(1,1)
# fig=plt.figure()
# ax=fig.add_subplot(111)
# x=np.arange(1,5)
# ax.plot(x,np.exp(x))
# ax.plot(x,x**2)
# #设置标题
# ax.set_title("Normal scale",fontdict=fontdict)
# # 设置x、y轴标签
# ax.set_xlabel("x axis", fontdict=fontdict)
# ax.set_ylabel("y axis", fontdict=fontdict)
# #a=plt.figure().add_subplot().set()
# #大概就是以上这个结构
# plt.show()

# Matplotlib 可以根据自变量与因变量的取值范围，自动设置 x 轴与 y 轴的数值大小。
# 当然，您也可以用自定义的方式，通过 set_xlim() 和 set_ylim() 对 x、y 轴的数值范围进行设置。
# fig, a1 = plt.subplots(1, 1)
# x=np.arange(1,10)
# a1.plot(x, np.exp(x), 'r')
# a1.set_title('exp')
# # 设置y轴
# a1.set_ylim(0, 4000)
# # 设置x轴
# a1.set_xlim(0, 8)
# plt.show()


#移动坐标轴以及为坐标轴添加箭头可以通过mpl_toolkits.axisartist实现
# 新建一个画板（画图视窗）
# fig = plt.figure('Sine Wave') #这里的是窗口名称
# # 新建一个绘图区对象ax,并添加到画板中
# ax=axisartist.Subplot(fig,1,1,1)
# fig.add_subplot(ax)
# # 隐藏默认坐标轴（上下左右边框）
# ax.axis[:].set_visible(False)
# #这里的axis就是指坐标轴
# # ax.axis["top"].set_visible(False)
# # ax.axis["right"].set_visible(False)
# # 新建可移动的坐标轴X-Y
# ax.axis['x']=ax.new_floating_axis(0,0)
# ax.axis['y']=ax.new_floating_axis(1,0)
# # new_fixed_axis(self, loc, offset=None)和new_floating_axis(self, nth_coord, value, axis_direction=‘bottom’)，
# # 而new_floating_axis()相对更加灵活。
# # （1）nth_coord：坐标轴方向，0代表X方向，1代表Y方向
# # （2）value：坐标轴处于位置，如果是平行与X轴的新坐标轴，则代表Y位置（即通过（0，value）），如果是平行与Y轴的新坐标轴，则代表X位置（即通过（value，0））。
# #这个value自己改一下参数就知道什么意思了
# # （3）axis_direction：代表刻度标识字的方向，可选[‘top’, ‘bottom’, ‘left’, ‘right’]
# # 设置刻度标识方向
# ax.axis["x"].set_axis_direction('top')
# ax.axis["y"].set_axis_direction('left')
# #我觉得上面这两行没什么用...
# # 加上坐标轴箭头，设置刻度标识位置
# ax.axis['x'].set_axisline_style("->",size=2.0)
# ax.axis["y"].set_axisline_style("->",size=2.0)
# # 画上y=sin(t)折线图，设置刻度范围，设置刻度标识，设置坐标轴位置
# t = np.linspace(0, 2 * np.pi)
# y = np.sin(t)
# ax.plot(t, y, color='red', linewidth=2)
# #我们可以设置标题的文本、字号和与坐标系顶端的距离（即上边距，用pad参数来控制）。
# ax.set_title('y = 2sin(2t)', fontsize=14, pad=20)
# #设置坐标标签
# ax.set_xticks(np.linspace(0.25, 1.25, 5) * np.pi)
# ax.set_yticks([-2,-1,0, 1, 2])
# #这俩是设置坐标轴的范围
# ax.set_xlim(-0.5 * np.pi, 1.5 * np.pi)
# ax.set_ylim(-2, 2)
# plt.show() 

#设置刻度和标签
#刻度指的是轴上数据点的标记，Matplotlib 能够自动的在 x 、y 轴上绘制出刻度。
#这一功能的实现得益于 Matplotlib 内置的刻度定位器和格式化器（两个内建类）。
#xticks() 和 yticks() 函数接受一个列表对象作为参数，列表中的元素表示对应数轴上要显示的刻度。
#例如ax.set_xticks([2,4,6,8,10])
# x=np.arange(0,math.pi*2,0.05)
# fig,ax=plt.subplots(1,1,figsize=(5,6))
# y=np.sin(x)
# ax.set_xlabel('angle')
# ax.set_ylabel('Sine')
# a=[0,2,4,6]
# ax.set_xticks(a)
# ax.set_xticklabels(['zero','two','four','six'],rotation=45)
# #其中rotation代表标签逆时针旋转的度数，这里指的是逆时针旋转45度
# ax.set_yticks([-1,0,1])
# ax.plot(x,y)
# plt.show()


#添加图例和标题
# 图例通过ax.legend或者plt.legend()实现
# 标题通过ax.set_title()或者plt.title()实现
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False
#以上这两行是设置字体，因为matplotlib默认不支持显示简体中文
#以上这里是引用了matplpotlib的rc函数，下文会解释
#也可以定义一个字体结构，再在设置标签和标题时具体指定
#fontdict = {'weight': 'normal', 'family': 'Times New Roman', 'size': 20}
fig=plt.figure()
ax=fig.add_subplot(111)
x=np.linspace(-2*np.pi,2*np.pi,200)
#生成一个闭区间内的等差数列，类型为ndarray，数量200
y=np.sin(x)
y1=np.cos(x)
ax.plot(x,y,label=r"$\sin(x)$")
ax.plot(x, y1, label=r"$\cos(x)$")
# ax.plot(x,y,label="$sin(x)$")
# ax.plot(x, y1, label="$cos(x)$")
#以上plot函数内的label为设置两个曲线的名称，以便生成图例
#这里使用了LaTex输出格式，
#在Python中，可以使用字符串前缀 r 或 R 来表示一个Raw字符串，
# Raw字符串会将反斜杠字符（\）视为普通字符而不是特殊的转义字符。这个特性可以方便地实现LaTeX格式化字符串的输出。
#在LaTeX中，通常使用$包裹起来的内容会被解析为一个数学公式。
# \是LaTeX的转义符号，用于对后面的符号进行转义，将其解析为LaTeX命令而不是普通字符。
#注意,这里虽然使用了r命令，但是后面的\没有被当作普通字符输出，这可能是因为在执行r之前，由于$的出现导致字符串先被传入一个LaTeX的编译器中，被当作LaTex命令执行，所以这里有没有r都一样，r没有起到作用，同时有没有\做LaTex的转义字符都一样。这里$的优先级可能高于另外两个命令。
#以上这句话是我结合chargpt自己理解的，不一定对
ax.legend(loc='best')
#图例传入一个位置loc
ax.set_title('正弦函数和余弦函数的折线图')
plt.show()




