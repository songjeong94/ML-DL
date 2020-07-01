#14681번
#사분면에서 나의 입력좌표값이 몇분면인지 맞춰보자

a = input()
b = input()
a = int(a)
b = int(b)

if a > 0 and b > 0:
    print("1")
elif a < 0 and b > 0:
    print("2")
elif a < 0 and b < 0:
    print("3")
else:
    print("4")
        

