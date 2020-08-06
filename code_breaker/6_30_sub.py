count = int(input())
count = range(count)
stack = 0
case = []

for i in count:
    classes = input().split()
    score = classes[1:]
    score = list(map(int, score))
    student = classes[0]
    student = list(map(int, student))
    avg = sum(score)/student[0]
    percent = float(100/student[0])
    for i in score:
        if i > avg:
            stack+=1
    ans = float(percent * stack)
    ans = round(ans, 3)
    stack = 0
    ans = "%.3f" %ans
    case.append(ans)

for i in case:
    print('{}%'.format(i))