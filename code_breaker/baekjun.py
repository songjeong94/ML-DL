M, F, K = map(int,input().split())

M = M - K

if M/2 > F:
    print(int(F))
elif M/2 < F:
    print(int(M/2))
elif M/2 == F:
    print(int(M/2))

