#环形队列的实现方式rear=(rear+1)%maxSize队尾指针+1
                #fornt=(front+1)%maxSize队首指针+1
#队空时rear=front
#队满时rear+1=front（要牺牲一个存储单位）
#队满要满足(rear+1)%maxSize=front
#一下这个建立队列的方法中默认第一个位置是没有元素的

class Queue:
    def __init__(self,size=100) -> None:  #列表大小为100
        self.queue=[0 for _ in range(size)]   #初始化时要指定列表长度
        self.rear=0   #队尾指针
        self.front=0    #队首指针
        self.size=size
            #此时队列为空，没有元素
    def push(self,element):
        #如果队满，则越界，无法再插入新元素
        if not self.is_filled():
            self.rear=(self.rear+1)%self.size    #队尾指针+1
            self.queue[self.rear]=element
        else:
            raise IndexError('Queue is filled')
    def pop(self):
        #如果队空，则不能弹出新元素
        if not self.is_empty():
            self.front=(self.front+1)%self.size
            return self.queue[self.front]
        else:
            raise IndexError('Queue is empty')
    #判断队空
    def is_empty(self):
        return self.rear==self.front
    #判断队满
    def is_filled(self):
        return (self.rear+1)%self.size==self.front
    
q=Queue(5)
for i in range(4):
    q.push(i)
for i in range(4):
    q.pop(i)


