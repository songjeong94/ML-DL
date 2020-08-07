

    
A, B, C = map(int, input().split())

count = 0
mul = 0

while 1:
    count += 1
    mul += A
    if count > 0 and count %7 == 0:
        mul = mul + B 
    if mul >= C:
        print(count)
        break



    
    
