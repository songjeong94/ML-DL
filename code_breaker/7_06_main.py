#가로n 세로n 의 좌석이 존재할 때 n범위를 볼 수 있는 cctv가 있다.
#이 때 모든좌석을 cctv로 확인할려면 몇대가 필요할까.

a,b,c = map(int, input().split())

if a%c == 0:
    a = int(a/c) 
else:
    a = int(a/c) + 1
#가로의 좌석과 cctv의 범위가 딱맞아지지 않을때 cctv1대를 더 추가한다

if b%c == 0:
    b = int(b/c)
else:
    b = int(b/c) + 1

#세로의 좌석과 cctv의 범위가 딱맞아지지 않을때 cctv1대를 더 추가한다.

camera = a * b 
#필요한 총 cctv의 대수
print(camera)