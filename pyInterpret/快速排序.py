#取一个原素p(第一个元素)，使元素p归位
#列表被p分成两部分，左边都比p小，右边都比p大
#递归完成排序
import random
from call_time import cal_time
import sys
sys.setrecursionlimit(10000)   #修改递归最大深度
 


def partition(li,left,right): #这个函数的作用是进行快速排序的第一步和第二步
    #把第一个元素存起来
    tmp=li[left]
    #从右边找比第一个元素小的元素放左边
    #从左边找比第一个元素大的元素放右边
    while left<right:
        while li[right]>=tmp and left<right:   #从右边找比左边tmp小的数
            #这里的left<right防止出现的情况：
            #右边的数全比第一个元素大，右指针等于左指针时仍然不能退出内层while循环的情况
            right-=1
            #如果右边第一个元素比tmp大，则tmp位置向左走一步
        #如果找到了比第一个元素小的数
        li[left]=li[right]  #把右边的值写道左边的空位上
        while li[left]<=tmp and left<right:
            left+=1
        li[right]=li[left]    #把左边的值写倒右边的空位上
    li[left]=tmp
    #两个指针相等时，把最初的tmp归位
    return left
    #返回最终的指针位置，为递归做准备


#@cal_time  #函数存在递归，这个无法判断总时间
def quick_sort(li,left,right):
    if left<right:    #至少两个元素
        mid=partition(li,left,right)
        quick_sort(li,left,mid-1)
        quick_sort(li,mid+1,right)
    return li
#递归对于python来说比较占内存
#时间复杂度为O(nlogn)

#可以把上面的快速排序封装成一个函数，计算这个函数的运行时间
@cal_time
def _quick_sort(li):
    quick_sort(li,0,len(li)-1)
    return li

li=list(range(10000))
random.shuffle(li)
print(_quick_sort(li))

li=list(range(10000,0,-1))
print(_quick_sort(li))
#最坏情况：倒叙列表，使用快速排序，每次只排了一个数，会导致时间复杂度退化成O(n^2)
#这里会超过递归最大深度，使用sys修改递归最大深度
#也可以先用random打乱，再用快速排序