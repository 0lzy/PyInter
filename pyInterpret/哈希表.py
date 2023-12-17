#哈希表是一个通过哈希函数来计算数据存储位置的数据结构，通常支持如下操作(模拟python的集合)
#insert(key,value):插入键值对（key，value）
#get（key)：如果存在键位key的键值对则返回其value，否则返回空值
#delete（key）：删除键为key的键值对

#直接寻址表：
#当域U很大时，需要消耗大量内存，很不实际
#如果U域很大而实际出现的key很少，则大量空间被浪费
#无法处理关键字不是数字的情况

#直接寻址表：key为k的元素放到k位置上
#改进直接寻址表:哈希（Hashing）
#构建大小为m的寻址表
#key为k的元素放到h(k)的位置上
#h(k)是一个函数，其将域U映射到表T[0,1,...,m-1]
#h(k)将原本的直接寻址的下表压缩

#哈希表（Hash Table,又称为散列表）,是一种线性存储结构。
#哈希表由一个直接寻址表和一个哈希函数组成。
#哈希函数h(k)将元素关键字k作为自变量，返回元素的存储下标

#哈希冲突：由于哈希表的大小是有限的，而要存储的值得总数量是无限的，因此对于任何哈希函数都会出现两个不同元素映射到同一个位置上的情况，这种情况叫做哈希冲突。
#开放寻址法：如果哈希函数返回的位置已经有值，则可以向后探查新的位置来存储这个值
#线性探查：如果i位置被占用，则探查i+1，i+2
#二次探查：如果i位置被占用，则探查i+i^2
#二度哈希：有n个哈希函数，当使用第1个哈希函数h1发生冲突时，则尝试使用h2.h3......

#拉链法解决哈希冲突
#哈希表的每一个位置都连接一个链表，当冲突发生时，冲突的元素将被加到该位置链表的最后

#常见哈希函数
#除法哈希法：h(k)=k%m
#乘法哈希法：h(k)=floor(m*(A*key%1))
#全域哈希法：h(k)=((a*key+b) mod p) mod m, a,b=1,2,...,p-1

class LinkList:  #链表
    class Node:  #链表中的节点
        def __init__(self,item=None):
            self.item=item
            self.next=None

    class LinkListIterator:   #一个迭代器类，让链表支持迭代
        def __init__(self,node):
            self.node=node
        def __next__(self):
            if self.node:
                cur_node=self.node
                self.node=cur_node.next
                return cur_node.item
            else:
                raise StopIteration
        def __iter__(self):
            return self

    def __init__(self,iterable=None):   #iterable传一个列表进来，可迭代的
        self.head=None              #最初链表头节点和尾节点都是空
        self.tail=None
        if iterable:               #如果iterable不是空的
            self.extend(iterable)    #调用extend循环插入

    def append(self,obj):      #插入一个元素（尾插），obj为要插入的元素
        s=LinkList.Node(obj)   #创建一个节点（头节点）
        if not self.head:     #如果链表为空，head和tail都是头节点
            self.head=s
            self.tail=s
        else:                #如果链表不为空，则先把尾巴接起来，再更新尾指针
            self.tail.next=s
            self.tail=s

    def extend(self,iterable):
        for obj in iterable:
            self.append(obj)

    def find(self,obj):   #查找函数
        for n in self:
            if n==obj:
                return True

    def __iter__(self):   #迭代函数，让链表支持迭代
        return self.LinkListIterator(self.head)

    def __repr__(self):   #重要函数，链表输出格式
        return "<<"+",".join(map(str,self))+">>"

lk=LinkList([1,2,3,4,5])
print(lk)
for element in lk:
    print(element)

#类似于集合的结构
class HashTable:
    def __init__(self,size=101):
        self.size=size
        self.T=[LinkList() for _ in range(self.size)]  #建立哈希表

    def h(self,k):   #哈希函数
        return k%self.size   #除法哈希

    def insert(self,k):
        i=self.h(k)    #哈希值
        if self.find(k):
            print("Duplicated Insert.")
        else:
            self.T[i].append(k)

    def find(self,k):  #查找
        i=self.h(k)
        return self.T[i].find(k)

ht=HashTable()
ht.insert(0)
ht.insert(1)
ht.insert(102)   #102和1在一起
print(','.join(map(str,ht.T)))   #打印哈希表
print(ht.find(102))   #查找102
print(ht.find(202))