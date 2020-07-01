#10039번 
# 5사람의 시험 점수의 평균을 구하자
# 40점 미만인 사람은 40점으로 한다.

students = range(5)
sum = 0
for i in students:
    score = int(input())
    if score < 40:
        score = 40
    sum += score
print(int(sum/5))

