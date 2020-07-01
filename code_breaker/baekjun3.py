num = int(input())
num1 = 2*num-1
num1 = range(num1)
num2 = 2*num-1
cro = 1

for i in num1:
    if i < num:
        i +=1
        num2 -= 2
        print(" "*i+"*"*num2+" "*i)
    # if i >= num:
    #     num2 += 2
    #     print("*"*num2)