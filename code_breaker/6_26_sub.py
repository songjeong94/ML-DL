c = int(4)
place = range(c)
tmp = 0
for i in place:
    sec = int(input())
    tmp = tmp + sec
print(int(tmp/60))
print(tmp%60)
    