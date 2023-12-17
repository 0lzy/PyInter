#拼接数字最大问题
#有n个非负整数，将其按照字符串拼接的方式拼接为一个整数，如何拼接可以使得得到得整数最大？
#例如：32，94，128，1286，6，71可以拼接成得最大整数为94716321286128
# a='96'
# b='87'
# if a>b:
#     a+b
# else:
#     b+a

# a='128'
# b='1286'
# # a+b='1281286'
# # b+a='1286128'
# if a+b>b+a:
#     a+b
# else:
#     b+a

# a='728'
# b'7286'
# # a+b='7287286'
# # b+a='7286728'

from functools import cmp_to_key

li=[32,94,128,1286,6,71]
def xy_cmp(x,y):
    if x+y<y+x:
        return 1
    elif x+y>y+x:
        return -1
    else:
        return 0

def num_join(li):
    li=list(map(str,li))   #把每个字符串转换成一个字符串然后组成一个新列表
    li.sort(key=cmp_to_key(xy_cmp))    #key为定义一个排序规则:如果前一个数拼接后一个数的和大于后一个数拼接前一个数的和，则让第一个数在前面
                                            #反之，让另一个数在后面，这里定义的排序方法也可以用冒泡等排序实现
    return ''.join(li)

print(num_join(li))
