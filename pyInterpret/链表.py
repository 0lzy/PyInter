#链表是由一系列节点组成的元素集合
#每个节点包含两个部分，数据域item和指向下一个节点的指针next
#通过节点之间的相互连接，最终串联成一个链表

#首先定义了一个节点类 Node，节点对象有两个属性：
#item 存储节点的值,next 存储指向下一个节点的引用。
#类似于c语言中的结构体类型
class Node:
    def __init__(self,item) -> None:
        self.item=item
        self.next=None

a=Node(1)
b=Node(2)
c=Node(3)
a.next=b
c.next=c
print(a.next.item)

#头插法
def create_linklist_head(li):
    head=Node(li[0])     #创建一个头节点
    for element in li[1:]:   
        node=Node(element)   #依次创建下一个节点
        node.next=head
        head=node
    return head

#尾插法
def create_linklist_tail(li):
    head=Node(li[0])     #创建一个头节点
    tail=head    #因为头节点要求不变，所以要额外维护一个尾节点让头节点保持不变
    for element in li[1:]:   
        node=Node(element)   #依次创建下一个节点
        tail.next=node
        tail=node
    return head

#打印链表（链表的遍历）
def print_linklist(lk):
    while lk:    #当链表非空时
        print(lk.item,end=',')
        lk=lk.next

lk=create_linklist_head([1,2,3])
print_linklist(lk)
print('')
lf=create_linklist_tail([1,2,3])
print_linklist(lf)


#链表节点的插入（先尾后头）
#p.next=curNode.next
#curNode.next=p

#节点链表的删除
#p=curNode.next
#curNode.next=p.Next
#del p

#双链表的每个节点有两个指针：一个指向后一个节点，另一个指向前一个节点。
#p.next=curNode.next
#curNode.next.prior=p   先尾
#p.prior=curNode
#curNode.next=p        后头
#处理顺序：对于插入节点p来说先尾后头

