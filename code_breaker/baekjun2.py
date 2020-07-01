a = int(input())
count = 0
a_first = str(a)


if 0 <= a <= 99: 
    if a > 10:
        first = a_first[0] + a_first[1]
        a1_int, a2_int = int(a_first[0]),int(a_first[1])
        while True:
            a_int_sum = (a1_int + a2_int)
            a2_str = str(a2_int)
            a_str_sum = str(a_int_sum)
            a_str_sum = a_str_sum[-1]
            a1_int,a2_int = int(a2_str), int(a_str_sum)
            count +=1
            if a2_str + a_str_sum == first:
                print(count)
                break
    elif a < 10:
        first2 = '0' + a_first[-1]
        a1 = '0'
        a1_int, a2_int = int(a1), int(a)
        while True:
            a_int_sum = (a1_int + a2_int)
            a2_str = str(a2_int)
            a_str_sum = str(a_int_sum)
            a_str_sum = a_str_sum[-1]
            a1_int, a2_int = int(a2_str), int(a_str_sum)
            count +=1
            if a2_str + a_str_sum == first2:
                print(count)
                break

