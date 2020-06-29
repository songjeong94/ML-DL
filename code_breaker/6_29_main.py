A,B = input().split()
A = int(A)
B = int(B)
if -10000 <= A and B <= 10000:
    if B < A:
        print(">")
    elif A < B:
        print("<")
    else: 
        print("==")