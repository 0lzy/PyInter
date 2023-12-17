#给一个二维列表，表示迷宫（0表示通道，1表示围墙）
#给出算法，求一条走出迷宫的路径

#深度优先搜索
#回溯思想
#思路：从一个节点开始，任意找下一个能走的点，当找不到能走的点时，退回上一个点寻找是否有其他方向的点
#使用栈存储当前路径
#在走迷宫的过程中，删除不可行的道路，保留最终可行的所有道路即为迷宫的可行路径
# import random
# maze=[[random.randint(0,1) for _ in range(10)] for _ in range(10)]
# print(maze)

maze=[[1,1,1,1,1,1,1,1,1,1],
      [1,0,0,1,0,0,0,1,0,1],
      [1,0,0,1,0,0,0,1,0,1],
      [1,0,0,0,0,1,1,0,0,1],
      [1,0,1,0,0,0,1,0,0,1],
      [1,0,0,0,1,0,0,0,0,1],
      [1,0,1,0,0,0,1,0,0,1],
      [1,0,1,1,1,0,1,1,0,1],
      [1,1,0,0,0,0,0,0,0,1],
      [1,1,1,1,1,1,1,1,1,1]]

#新建一个列表，用来存储方向
dirs=[
    lambda x,y:(x+1,y),   #上
    lambda x,y:(x-1,y),   #下
    lambda x,y:(x,y-1),   #左
    lambda x,y:(x,y+1)    #右
]

def maze_path(x1,y1,x2,y2):   #x1,y1代表起点位置，x2,y2代表终点位置
    stack=[]     #建栈
    stack.append((x1,y1))   #一开始栈中只有起点坐标
    while(len(stack)>=0):    #栈不为空
                            #如果栈为空，则两个点之间没有通路，说明迷宫里有一个区域四面全是墙
        curNode=stack[-1]   #当前位置为栈的最后一个元素
        if curNode[0]==x2 and curNode[1]==y2:
            #走到终点了
            for p in stack:
                print(p)
            return True
        #x,y四个方向(x-1,y),(x+1,y),(x,y-1),(x,y+1)
        for dir in dirs:
            nextNode=dir(curNode[0],curNode[1]) #下一个位置坐标
            #一个0一个1的原因是，curNode中只有两个元素，一个x，另一个是y
            #这里的dir为dirs里封装的lambda函数
            #如果下一个位置能走
            if maze[nextNode[0]][nextNode[1]]==0:   #0表示能走
                stack.append(nextNode)
                maze[nextNode[0]][nextNode[1]]=2  #2表示为已经走过
                break
        else:
            maze[nextNode[0]][nextNode[1]]=2
            stack.pop()
    else:
        print("无路可走")
        return False

maze_path(1,1,8,8)
