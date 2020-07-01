# 5543번
# 상근 날드의 메뉴 세트중에 어떤것이 제일쌀까
# 세트로 구매시 50원 할인!
bmenu = range(3)
dmenu = range(2)

burger = []
drink = []
for i in bmenu:
    price = int(input())
    burger.append(price)

for i in dmenu:
    price = int(input())
    drink.append(price)

b = min(burger)
d = min(drink)

print(b+d - 50)