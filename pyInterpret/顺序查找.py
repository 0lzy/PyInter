def linear_search(li,val):
    for ind,v in enumerate(li):
        if v==val:
            return ind 
        #具体地说，每次迭代会将 li 中的一个元素赋值给变量 v，同时 enumerate(li) 会返回一个元组 (ind, v)，
        #其中 ind 是这个元素在列表中的下标（从 0 开始计数）
    else:
        return None
