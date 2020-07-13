num = range(10)
aaa = []

for i in num:
    a = int(input())
    c = (a % 42)
    aaa.append(c)
aaa = list(set(aaa))
print(len(aaa))