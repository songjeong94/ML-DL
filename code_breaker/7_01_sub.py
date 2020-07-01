#2742번
# 자연수n을 지정했을때 n ~ 1까지 차례로 출력해보자!

n = int(input())
n2 = range(n)

for i in n2:
    print(n)
    n -= 1