#从斐波那契数列看动态规划
#F(n)=F(n-1)+F(n-2)
#练习：试用递归和非递归的方法来求解斐波那契数列的第n项

#递归
def fibnacci(n):
    if n==1 or n==2:
        return 1
    else:
        return fibnacci(n-1)+fibnacci(n-2)
    
print(fibnacci(10))


def fibnacci2(n):
    if n==1 or n==2:
        return 1
    else:
        fibnacci_list=[1,1]
        for i in range(2,n):
            fibnacci_list.append(fibnacci_list[i-1]+fibnacci_list[i-2])
        return fibnacci_list[n-1]
    
print(fibnacci2(10))
