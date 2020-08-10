count = int(input())
count = range(count)
stack = 0


for i in count:
    classes = list(map(int,input().split()))
    score = classes[1:]
    student = classes[0]
    avg = sum(score)/student[0]
    percent = float(100/student[0])
    for i in score:
        if i > avg:
            stack+=1
    ans = float(percent * stack)
    ans = round(ans, 3)
    stack = 0
    ans = "%.3f" %ans
    print('{}%'.format(ans))

