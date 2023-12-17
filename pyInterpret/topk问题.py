#现在有n个数，设计算法得到前k大的数（k<n)
#解决思路：
    #排序后切片：O(nlogn)
    #冒泡，选择，插入：O(kn)
#堆排序思路：O(nlogk)(每调整一次为logk)
    #取列表前k个元素建立一个小根堆，堆顶就是目前第k大的数，也是目前小根堆中最小的数
    #然后依次向后遍历列表，对于列表中的元素，如果小于堆顶，则忽略该元素；
    #如果大于堆顶元素，则将堆顶更换为该元素，并且对堆进行一次调整
    #遍历列表所有元素后，倒叙弹出堆顶

#小根堆调整
def sift(li,low,high):  #向下调整函数
    #假设除了他自己外，其他子树全是堆
    """
    li:列表
    low：堆的根节点位置
    high：堆的最后一个元素位置（！也可以是整个堆的最后一个元素 ）
    """
    i=low   #最开始指向根节点
    j=2*i+1  #为i的左孩子
    tmp=li[low]  #把根节点存起来，i是变的，这里不能写li[i]
    while j<=high and j+1<=high:   #只要j这个位置有数，就一直循环（不要让j超过最后一层）
                                #保证有右孩子，防止越界
        if li[j+1]<li[j]:
            j=j+1    
        if li[j]<tmp:
            li[i]=li[j]  #j的位置为空了
            i=j        #往下看一层
            j=j*2+1    #孩子也跟着往下
        else: #li[j]<=tmp
            li[i]=tmp   #把tmp放在某一级领导的位置上
            break
    else:
        li[i]=tmp  #如果j越界了，i就是指最后一层了，把tmp放到叶子节点上

def topk(li,k):
    heap=li[0:k]  #取列表的前k个元素
    for i in range((k-2)//2,-1,-1):
        sift(heap,i,k-1)   #建堆
    for i in range(k,len(li)-1):
        if li[i]>heap[0]:
            heap[0]=li[i]
            sift(heap,0,k-1) #如果换了堆顶，则做一次调整
    #排序输出
    for i in range(k-1,-1,-1): 
        #i指向堆的最后一个元素
        heap[0],heap[i]=heap[i],heap[0]  #让棋子上去（去掉堆顶）
        #heap是堆，而li不是堆，这里要注意不能写li[i]和sift(li,0,i-1)
        sift(heap,0,i-1)   #通过调整重新让其成为堆（不过这里high是i-1，因为第i个位置已经存储了最大的元素）
    print(heap)


li=list(range(1000))
import random
random.shuffle(li)
topk(li,5)


# def heap_sort(li):
#     #如果孩子的下标是i（不管是左孩子还是右孩子），那么他的根节点下标就是（i-1）//2
#     #如果线性存储的二叉树，他的最后一个叶子节点的下标为n-1，那么他的根节点的下标就是(n-2)//2
#     n=len(li)
#     #堆排序从最底层开始，最底层堆的根节点下标为（n-2)//2
#     for i in range((n-2)//2,-1,-1):
#         #i代表建堆的时候调整的部分下标根的下标
#         sift(li,i,n-1)
#     #建堆完成了
#     print(li)
#     for i in range(n-1,-1,-1): 
#         #i指向堆的最后一个元素
#         li[0],li[i]=li[i],li[0]  #让棋子上去（去掉堆顶）
#         sift(li,0,i-1)   #通过调整重新让其成为堆（不过这里high是i-1，因为第i个位置已经存储了最大的元素）
#     print(li)

