day = int(input())
count = 0
number = []
aaa = map(int,input().split())
number.append(aaa)
print(number)
print(list(number))

for i in number:
    print(i)
    if i == day:
        count += 1
print(count)


