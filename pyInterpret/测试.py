n = 521
c=str(n)
length=len(c)
print(length)
s=list()
sum=0
for i in range(length):
    a=int(c[i])
    if i%2==0:
        sum=sum+a
    else:
        sum=sum-a
print(sum)