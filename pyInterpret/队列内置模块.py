from collections import deque

# q=deque([1,2,3,4,5],5) #建队,第二个参数为队列长度
# #队列满，队首自动出队
# q.append(6)       #单向队列队尾进队
# print(q.popleft())  #单向队列队首出队

# #双向队列
# q.appendleft(1)   #队首进队
# print(q.pop())    #队尾出队

def tail(n):
    with open('linux tail.txt','r') as f:
        q=deque(f,n) 
        return q
for line in tail(5):
    print(line,end='')
