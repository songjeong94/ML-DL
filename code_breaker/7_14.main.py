a = range(3)
aaa = []

for i in a:
    a = input().split()
    aaa.extend(a)
c1 = aaa.count(max(aaa))
c2 = aaa.count(min(aaa))

if c1 > c2:
    print(min(aaa), max(aaa))
else:
    print(max(aaa), min(aaa))

    

