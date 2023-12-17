#多关键字排序；加入现在有一个员工表，要求按照薪资排序，年龄相同的员工按照年龄排序
    #先按照年龄进行排序，再按照薪资进行稳定的排序
#对32，13，94，52，17，54，93排序，是否可以看成多关键字排序
#如果先看十位数，再看个位数的话，可以看成有两个关键字的多关键字排序

#基数排序则是从个位开始排，再看十位......
#类似于桶排序,但也有区别
def radix_sort(li):
    max_num=max(li)        #循环的次数根据最大值的位数确定
                            #99->2,999->3,9999->4
    it=0      #it代表迭代多少次
    while 10**it<=max_num:
        buckets=[[] for _ in range(10)]  #固定分十个桶,因为是要根据位数来分桶的，一个位上的数字只有0~9这10个数字
        for var in li:
            #987 it=1  987//10->98 98%10->8
                #it=3  987//100->9  9%10->9
            digit=(var//10**it)%10  #取这一次迭代时的位数，it=0时取个位，it=1时取十位......
            buckets[digit].append(var)
        #分桶完成
        li.clear()
        for buc in buckets:
            li.extend(buc)
        #把数重新写回li
        it+=1      
    return li
#时间复杂度O(kn)  k表示while执行了几次

import random
li=list(range(1000))
random.shuffle(li)
radix_sort(li)
print(li)
        

