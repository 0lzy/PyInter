#给定一个字符串，其中包含小括号、中括号、大括号，求该字符串中的括号是否匹配

#先让一个括号进栈，再看下一个括号，看看是否与栈顶匹配
#如果匹配，则栈顶出栈
#如果不匹配，则进栈
#如果读完字符之后，栈清零了，则括号是匹配的

class Stack():
    def __init__(self):
        self.stack=[]
    def push(self,element):
        self.stack.append(element)
    def pop(self):
        return self.stack.pop()
    def get_top(self):
        if len(self.stack)>0:
            return self.stack[-1]
        else:
            return None
    def is_empty(self):
        return len(self.stack)==0
        

def brace_match(s):
    match={'}':'{',']':'[',')':'('}
    stack=Stack()
    for ch in s:
        if ch in {'(','[','{'}:
            stack.push(ch)
        else:   #ch in 右括号
            if stack.is_empty():
                return False
            elif stack.get_top()==match[ch]:
                stack.pop()
            else:  #stack.get_top!=match[ch]
                return False
    if stack.is_empty():
        return True
    else:    #出现左右括号数量不匹配的情况
        return False
    
print(brace_match('({)}{)()})'))
print(brace_match('[({[]})]'))


            