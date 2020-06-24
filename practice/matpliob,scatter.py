import matplotlib.pyplot as plt
import numpy as np

# plt.plot([1,2,3,4]) 
#plt.show()

# plt.plot([2,3,4],[4,9,16], c ='yellow')
# plt.title('practice',fontsize=10)
# plt.xlabel('x datas')
# plt.ylabel('y datas')
#plt.show()

# plt.plot([2,3,4],[4,9,16], c='black')
# plt.axis([0,10,0,20])
# plt.grid() #  눈금 칸 표현
#plt.show()

# t = np.arange(0, 10, 2)
# plt.plot(t, t, 'r--') # --는 대쉬를 의미
# plt.plot(t, t, 'rs') # s는 사각형 포인트
# plt.plot(t, t**2, 'b--')
# plt.plot(t, t**3, 'b^') #^는 삼각형을 의미
# plt.plot(t, t**3, 'g--')
# plt.plot(t, t**3, 'go') #o는 동그라미 표현
# plt.show()

# plt.plot(t,t, 'rs--')
# plt.plot(t,t**2, 'b^--')
# plt.plot(t, t**3, 'go--')
#이런식으로 합칠수 있다.
# plt.plot(t, t, 'rs--', t,t**2, 'b^--',t,t**3, 'go--')
#이런식으로 한번에 표현도 가능하다.

##############Scatter##################

# plt.scatter(3,5) #3,5에 점찍기
# plt.show()

x = np.arange(0, 10, 1)
plt.scatter(x, x**2, c='blue', s=10) #s를 이용해서 점크기 조정
plt.text(3, 50,'y=x^2 graph') #임의로 원하는 좌표에 텍스트 추가하기
plt.show()
