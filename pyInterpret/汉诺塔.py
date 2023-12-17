def hanoi(n,a,b,c):  #n表示n个盘子，a,b,c..表示这些盘子的名字
    #总目标：把n个盘子从a经过b移动到c
    if n>0:
        t=0
        hanoi(n-1,a,c,b)
        #先把上面的n-1个盘子从a经过c移动到b
        #再把最底下的第n个盘子从a直接移动到c
        print("moving from %s to %s"%(a,c))
        hanoi(n-1,b,a,c)
        #把第一步移动完的n-1个盘子从b经过a移动到c

hanoi(3,'A','B','C')
#汉诺塔移动次数递推式：h(x)=2h(x-1)+2