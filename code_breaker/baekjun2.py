sub = int(input())
sub =range(sub)
a = []

score = map(int,input().split())
a.append(score)

a.sort(reverse=True)
print(a)