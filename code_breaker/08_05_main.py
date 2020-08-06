N = int(input())
N = range(N)
mul = 1
ans = 1 

for i in N:
    i+= 1
    mul = 1 * i
    ans *= i
print(ans)