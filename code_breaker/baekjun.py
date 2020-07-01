n = int(input())
c = n
count = 0

while True:
    a = c//10
    b = c%10
    c = b*10+(a+b)%10
    count += 1
    print(count)
    print(c)
    if c == n:
        break
print(count)
