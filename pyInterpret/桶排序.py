#在计数排序中，如果元素的范围比较大（比如在1到1亿之间），如何改造算法
#桶排序（Bucket_sort）:首先将元素分在不同的桶中，在对每个桶中的元素排序

def bucket_sort(li,n=100,max_number=10000):
    '''
    n:表示分成几个桶
    max_number:列表中最大元素（要知道范围）
    '''
    buckets=[[] for _ in range(n)]
    #开一个二维列表，里面的一维列表就表示桶
    #创建n个空桶
    for var in li:
        a=max_number//n #一个同理放几个数
        i=min(var//a,n-1)    #i表示var放到几号桶里
        #取小的意思是：如果列表末尾有个别的数放入时超出桶的个数，也把他们放在最后一个桶里
        buckets[i].append(var)  #把原列表中的元素放入对应的桶
        #append是把元素放在列表的最后，所以要反向冒泡（桶里加一个元素就排一次，在加的时候就保持桶内的顺序）
        #（这里也可以用其他的排序方法）
        for j in range(len(buckets[i])-1,0,-1):
            if buckets[i][j]<buckets[i][j-1]:
                buckets[i][j],buckets[i][j-1]=buckets[i][j-1],buckets[i][j]
            else:
                break
    #所有的桶都建完了
    #以下是输出
    sorted_li=[]
    for buc in buckets:
        sorted_li.extend(buc)  #添加桶中的元素，展成一维
    return sorted_li

#桶排序的表现取决于数据的分布，对不同数据采取不同的分桶策略
#平均时间复杂度O(n+k)
#最坏时间复杂度O((n^2)k)
import random
li=[random.randint(0,10000) for _ in range(10000)]
li=bucket_sort(li)
print(li)

        