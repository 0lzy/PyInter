from call_time import cal_time

@cal_time  #对函数加装饰器
def linear_search(li,val):   #线性查找
    for ind,v in enumerate(li):
        if v==val:
            return ind 
        #具体地说，每次迭代会将 li 中的一个元素赋值给变量 v，同时 enumerate(li) 会返回一个元组 (ind, v)，
        #其中 ind 是这个元素在列表中的下标（从 0 开始计数）
    else:
        return None
@cal_time
def binary_search(li,val):  #二分查找
    left=0
    right=len(li)-1
    while(left<=right):     #候选区有值
        mid=(left+right)//2   #双//为整除
        if li[mid]==val:
            return mid
        elif li[mid]>val:    #待查找的值在mid左侧
            right=mid-1
        else:  # li[mid]<val: #待查找的值在mid右侧
            left=mid+1
    else:
        return None

li=list(range(1000000))
#print(binary_search(li,3))
print(linear_search(li,38900))
print(binary_search(li,38900))
#可见二分查找的时间接近0s
