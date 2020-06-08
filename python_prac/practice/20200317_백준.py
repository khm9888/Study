# 강민이는 치킨 한 마리를 주문할 수 있는 치킨 쿠폰을 n장 가지고 있다. 이 치킨집에서는 치킨을 한 마리 주문할 때마다 도장을 하나씩 찍어 주는데, 도장을 k개 모으면 치킨 쿠폰 한 장으로 교환할 수 있다.

# 강민이가 지금 갖고 있는 치킨 쿠폰으로 치킨을 최대 몇 마리나 먹을 수 있는지 구하여라. 단, 치킨을 주문하기 위해서는 반드시 치킨 쿠폰을 갖고 있어야 한다.
# import sys

# for i in sys.stdin.readlines():
# 	v_list=list(map(int, i.strip().split()))

# 	count=v_list[0]
# 	check=0

# 	while v_list[0]>=v_list[1]:
# 		v=v_list[0]//v_list[1]
# 		count+=v
# 		v_list[0]=v_list[0]%v_list[1]+v
# 	print(count)

num, base = map(int, input().strip().split(' '))
answer=0

for i,j in enumerate(num):
	print(i)
