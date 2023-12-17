#背包问题
#一个小偷在某个商店发现有n个商品，第i个商品价值vi元，重wi千克。
#他希望拿走的价值尽量高，但他的背包最多只能容纳w千克的东西。
#他应该拿走哪些商品？

#0-1背包：对于一个商品，小偷要么把它完整拿走，要么留下。不能只拿走一部分，或者把一个商品拿走多次。（商品为只因条）
#分数背包：对于一个商品，小偷可以拿走其中任意一部分。（商品为只因砂）
#0-1背包可能不能用贪心算法来解决，因为在拿去商品时可能出现全部拿取之后背包不满的情况
#所以以下为分数背包
goods=[(60,10),(100,20),(120,30)]
#每个商品元组表示（价格，重量）
goods.sort(key=lambda x: x[0]/x[1],reverse=True)  #价格除以重量，1千克值多少钱

def fractional_backpack(goods,w):   #w为背包所能承受的最大重量
    print(goods)
    m=[0 for _ in range(len(goods))]
    total_v=0
    for i,(price,weight) in enumerate(goods):
        if w>=weight:   #如果最贵的物品重量小于背包所能承受的重量，则全部拿走为1
            m[i]=1      
            w=w-weight
            total_v+=price
        else:          #如果最贵的物品重量大于背包所能承受的重量，则只能拿一部分为,比例为w/weight
            m[i]=w/weight   
            w=0
            total_v+=m[i]*price
            break
        
    return total_v,m
m=fractional_backpack(goods,50)
print(m)