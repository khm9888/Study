
import random as r

c = []
n=10
for i in range(n):
	c.append(0)

for i in range(1000):
	rand = r.randrange(0,n)
	c[rand]+=1

for i in range(len(c)):
	#print(type(c[i]/1000*100))
	print("%d 입니다" %int(c[i]/1000*100))
	print(c)

#  len(문자열, 리스트, 딕셔너리, )

dic = {}

dic[('key','k2')] = 'value1','v2'
print(dic)

for x in range(1,11):
	if x<4:
		print("탈락")
	elif x<7:
		print("보류")
	else :
		

