def count_sort(li,max_count=100):
    #给出一个元素大小在0到100范围内的列表
    count=[0 for _ in range(max_count+1)]
    #'_'表示无论我遍历到第几次，我不关心当前值是多少
    #这样写表示：生成一个长度为（max_count+1）,值全部为0的列表
    for val in li:
        count[val]+=1
        #遍历原列表中的元素，如果该元素出现一次，则以这个元素为下标，在count把该元素的出现次数加一
    li.clear()  #把li清空
    for ind,val in enumerate(count):
        #这里val是该元素在原列表li中出现的次数，ind则是原列表中的元素值
        #这里的val不是上面那个循环中的val
        for i in range(val):
            li.append(ind)  #ind在出现了val次，往清空后的li中加val个ind
    print(li)

#时间复杂度O(n)
#限制数有范围，要知道列表中的最大值，要求这些数全是正整数
#比较费空间

import random
li=[random.randint(0,100) for _ in range(1000)]
#随机生成一个0到100范围内的1000个数
count_sort(li)
        
