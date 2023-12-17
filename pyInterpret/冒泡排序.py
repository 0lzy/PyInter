#目的：排出一个升序的列表
#冒泡排序
#列表每两个相邻的数，入伙前面的比后面大，则交换这两个数
#一趟排序完成后，则无序区减少一个数，有序区增加一个数
#需要n-1(趟)次循环
import random
 
def bubble_sort(li):
    for i in range(len(li)-1):  #第i趟 (从0开始第1趟)
        exchange=False
        #无序区域:[0:n-1-i]
        for j in range(len(li)-i-1): #每一趟多少无序区域
            if li[j]>li[j+1]:
                li[j],li[j+1]=li[j+1],li[j]  #py独有的交换两个变量值
                exchange=True
        if exchange==False:    #如果在某一次循环中没有发生变量的交换，则我们认为已经排序完成，不需要进行接下来的继续循环
            return li
    return li
    

#降序
def bubble_sort_de(li):
    for i in range(len(li)-1):  #第i趟 (从0开始第1趟)
        #无序区域:[0:n-1-i]
        exchange=False
        for j in range(len(li)-i-1): #每一趟多少无序区域
            if li[j]<li[j+1]:
                li[j],li[j+1]=li[j+1],li[j]  #py独有的交换两个变量值
                exchange=True
        if exchange==False:
            return li
    return li

li=[random.randint(0,1000) for i in range(1000)]
print(li)
print(bubble_sort(li))
print(bubble_sort_de(li))

#时间复杂度O(n^2)
#原地排序，不额外占用空间