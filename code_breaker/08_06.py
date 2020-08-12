
count = 0
cc = []
for i in range(5):
    a = list(map(int,input().split()))
    count+=1
    score = sum(a)
    cc.append(score)
print("{} {}".format((cc.index(max(cc))+1), max(cc)))
