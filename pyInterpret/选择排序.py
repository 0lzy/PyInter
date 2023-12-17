def select_sort_simple(li):
    #新建一个列表，又占了一个空间，空间复杂度高
    li_new=[]  #建立一个新列表
    for i in range(len(li)):  #循环次数
        for val in li:
            min_val=min(li)
            li_new.append(min_val)
            li.remove(min_val)
    return li_new

def select_sort(li):
    #一趟排序记录最小的数，放到第一个位置
    #再一趟排序记录列表无序区最小的数，放到第二个位置
    #.......
    #算法关键点：有序区和无序区，无序区最小数的位置
    for i in range(len(li)-1): #循环次数
        min_loc=i
        #无序区长度
        for j in range(i,len(li)):
            if li[j]<li[min_loc]:
                min_loc=j
        #if min_loc!=i:
        li[i],li[min_loc]=li[min_loc],li[i]
    return li
#原地排序,节省内存
#时间复杂度O(n^2)

li=[3,5,2,1,7,8,4,9]
print(select_sort_simple(li))
li=[3,5,2,1,7,8,4,9]
print(select_sort(li))



