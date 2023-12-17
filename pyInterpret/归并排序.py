#已知一个列表，假设其中有两段是分别排序完的
#真正排序的列表没有这个假设
#例如：2，5，7，8，9，|1，3，4，6
def merge(li,low,mid,high):  #归并特性
    '''
    li:列表
    low:第一段第一个元素
    mid:第一段的最后一个元素
    high:第二段最后一个元素
    '''
    tmp=[]
    i=low   #第一段的第一个元素
    j=mid+1  #第二段的第一个元素
    while i<=mid and j<=high:
        #只要左右两边都有数
        if li[i]<li[j]:
            tmp.append(li[i])
            i+=1
        else:
            tmp.append(li[j])
            j+=1
    #while执行完后，肯定有一部分没有数了
    while i<=mid:
        tmp.append(li[i])
        i+=1
    while j<=high:
        tmp.append(li[j])
        j+=1
    #以上两个有一个执行
    li[low:high+1]=tmp
    return li

#分解：将列表越分越小，直至分成一个元素，一个元素是有序的
#终止条件：一个元素是有序的
#合并：将两个有序列表（最开始每个有序列表只有一个元素）归并，列表越来越大
def merge_sort(li,low,high):
    if low<high:   #至少有两个元素，因为只有一个元素的时候是low=high
        mid=(low+high)//2
        merge_sort(li,low,mid)   #左边归并排序
        merge_sort(li,mid+1,high)   #右边归并排序
        #不断递归会使原列表被拆分成一个元素
        merge(li,low,mid,high)   #使用归并特性使两边合并

    return li


def merge_sort_test(li,low,high):  #逐步检验
    if low<high:   #至少有两个元素，因为只有一个元素的时候是low=high
        mid=(low+high)//2
        merge_sort_test(li,low,mid)   #左边归并排序
        merge_sort_test(li,mid+1,high)   #右边归并排序
        #不断递归会使原列表被拆分成一个元素
        print(li[low:high+1])



#时间复杂度O(nlogn)

li=[2,4,5,7,1,3,6,8]
print(merge(li,0,3,7))
li_=list(range(100))
import random
random.shuffle(li_)
print(merge_sort(li_,0,99))
merge_sort_test(li_,0,len(li_)-1)
