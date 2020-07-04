# 10797번 10부제
# 자동차 10부제를 지키않는 차가 몇대일까!

day = int(input())
count = 0

aaa = map(int,input().split())

for i in aaa:
    if i == day:
        count += 1
print(count)


