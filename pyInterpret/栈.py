class Stack:
    def __init__(self):
        self.stack=[]     #初始化一个列表
    
    def push(self,element):
        self.stack.append(element)   #进栈

    def pop(self):
        return self.stack.pop()    #出栈
    
    def get_top(self):         #返回栈顶元素
        if len(self.stack)>0:
            return self.stack[-1]
        else:
            return None
        
stack=Stack()
stack.push(1)
stack.push(2)
stack.push(3)
print(stack.pop())
print(stack.get_top())
