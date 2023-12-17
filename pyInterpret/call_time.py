import time

#装饰器，测量一个函数运行的时间
def cal_time(func):
    def wrapper(*args,**kwargs):
        t1=time.time()
        result=func(*args,**kwargs)
        t2=time.time()
        print("%s running time is: %ssecs."%(func.__name__,t2-t1))
        return result
    return wrapper 