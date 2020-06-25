#=======10430번 문제풀이=======#
A,B,C = map(int, input().split())
# map을 사용하여 split의 결과를 한번에 정수형으로 변환
if 2 <= A or B or C <= 10000: 
    print((A+B)%C)
    print(((A%C)+(B%C))%C)
    print((A*B)%C)
    print(((A%C)*(B%C))%C)

# 출제형식에 있던 입력값의 조건과 출력