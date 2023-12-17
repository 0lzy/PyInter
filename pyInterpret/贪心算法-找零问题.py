#贪心算法（又称贪婪算法）是指，在对问题求解时，总是做出在当前看来最好的选择。
#也就是说，不从整体最优上加以考虑，他所做出的是在某种意义上的局部最优解
#贪心算法并不保证会得到最优解，但是在某些问题上贪心算法的解就是最优解，要会判断一个问题能否用贪心算法来计算。

#找零问题
t=[100,50,20,10,5,1]
def change(n,t):  #n是要找的钱数
    m=[0 for _ in range(len(t))]   #每种钱币要找几张
    #假设t是倒叙排序好的
    #t=sorted(t,reverse=True)
    t.sort(reverse=True)
    for i,money in enumerate(t):
        m[i]=n//money
        n=n%money  #n为每次找完大额面值后还剩的需要找的钱数
    return m,n

print(change(381,t))

