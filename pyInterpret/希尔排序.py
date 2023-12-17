#希尔排序是一种或分组插入排序算法
#首先取一个整数d1=n//2,将元素分为d1个组，每组相邻量元素之间距离为d1，在各组内进行直接插入排序
#取第二个整数d2=d1//2，重复上述分组排序过程，直到di=1,即所有元素在同一组内进行直接插入排序
#希尔排序每趟并不使某些元素有序，而是使整体数据越来越接近有序；最后一趟排序使得所有数据有序

#同插入排序,把所有的1改成gap,这样就是在每个组中分别进行插入排序
def insert_sort_gap(li,gap):
    #gap是每个组元素之间的间隔，也就是d的值
    for i in range(gap,len(li)):
        #i表示摸到的牌的下标
        temp=li[i]  #把摸到的牌存起来(缓存)
        j=i-gap  #j指的是手里牌的下标
        while li[j]>temp and j>=0:
            #while的作用是判断手牌是否需要向右移动
            #j>=0防止出现-1的情况
            li[j+gap]=li[j]
            j-=gap 
        li[j+gap]=temp
        
    return li

def shell_sort(li):
    d=len(li)//2
    while d>=1:
        insert_sort_gap(li,d)
        d//=2
    return li
#比单独的插入排序快的很多
#比堆排序慢
#时间复杂度和gap有关，比较复杂

li=list(range(100))
import random
random.shuffle(li)
print(li)
print(shell_sort(li))