num = int(input())
num = range(num)
a = []

for i in num:
    ch = str(input())
    a.append(ch)
print(a)
for i in a:
    for i in a[i]:
        i += 1
        'o'*i = 1*i     