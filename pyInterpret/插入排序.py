#初始时手里（有序区）只有一张牌
#每次（从无序区）摸一张牌，插入到有牌的正确位置
from call_time import cal_time
import random

@cal_time
def insert_sort(li):
    for i in range(1,len(li)):
        #i表示摸到的牌的下标
        temp=li[i]  #把摸到的牌存起来(缓存)
        j=i-1  #j指的是手里牌的下标
        while li[j]>temp and j>=0:
            #while的作用是判断手牌是否需要向右移动
            #j>=0防止出现-1的情况
            li[j+1]=li[j]
            j-=1  #把j的指针向前移动一个单位,j=-1则跳出while
            #将j依次减1，手牌不需要向右移动为止
        #直到移动停止后，记录j的位置，让j的下一个位置=待插入元素
        li[j+1]=temp
        
    return li
#原地排序
#时间复杂度O(n^2)


li=list(range(10000))
random.shuffle(li)#打乱
print(insert_sort(li))

