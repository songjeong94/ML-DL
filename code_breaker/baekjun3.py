
num = int(input())
num1 = 2*num-2
num1 = range(num1)
num2 = 2*num-1
num3 = num-1
cro = 1
res = num-1

for i in num1:
    if i < num:
        print(" "*i + "*"*num2)
        i +=1
        num2 -= 2 
    if i >= num:
        i += 11
        cro += 2
        res-=1
        print(" "*res + "*"*cro)