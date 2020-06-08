# korean=92
# english=47
# mathematics=86
# science=81

# print(korean >= 50 and english >= 50 and mathematics >= 50 and science >= 50
# )

#블랙잭

# N,M = map(int,input().split())

# values=list(map(int,input().split()))
# cnt=0
# max=float('-inf')
# for i in range(N):
# 	for j in range(1+i,N):
# 		for k in range(1+j,N):
# 			cnt+=1
# 			# print("i:%d,j:%d,k:%d,cnt:%d" %(i,j,k,cnt))
# 			v=values[i]+values[j]+values[k]
# 			if max<v and v<=M:
# 				max=v
# 			# print(max,"max")
# # print("cnt",cnt)
# # print("result",max)
# print(max)

# def P(n,m,c):
# 	t=set()
# 	for i in range(n-2):
# 		for o in range(i+1,n-1):
# 			for p in range(o+1,n):
# 				s=c[i]+c[o]+c[p]
# 				if s<=m:
# 					t.add(s)
# 					break

# 	return max([*t])
# print(P(*map(int,input().split()),list(sorted(map(int,input().split()))[::-1])))

# n=int(input())
# check=False
# predictMin = n-(len(str(n)*9))
# predictMin = 0 if predictMin < 0 else predictMin

# for i in range(predictMin,n+1):
# 	ans=i+sum(list(map(int,str(i))))
# 	if n==ans:
# 		check=True
# 		print(i)
# 		break
# if not check:
# 	print(0)

# N=int(input())
# v_list=list()
# for j in range(100):
# 	for i in range(1000):
# 		if j==0:
# 			v_list.append(int(str(i)+"666"))
# 		else:	
# 			v_list.append(int(str(i)+"666"+str(j)))
# v_list.sort()
# print(v_list[N])

# n = int(input())

# m=10000

# number = 666

# v_list=[]

# while m > 0:
#     if "666" in str(number):
#         m -= 1
#         v_list.append(number)
#     if m == 0: 
#         break
#     number += 1

# print(v_list[n-1])

import random

def is_bust(score):
	if score>21:
		bust=True
	else:
		bust=False
	return bust

def calculate(card_1,card_2):
	if type(card_1)==str:
		if card_1=="J" or card_1=="Q" or card_1=="K":
			card_1=10
		elif card_1=="A":
			card_1=[1,11]
	if type(card_2)==str:
		if card_2=="J" or card_1=="Q" or card_1=="K":
			card_2=10
		elif card_2=="A":
			card_2=[1,11]



def simulation(player,dealer,iteration_number=10000):

	result_list=list()
	for iterations in range(iteration_number):

		deck = ["A",2,3,4,5,6,7,8,9,"J","Q","K"]
		deck *= 4"*8
		player_score=0
		dealer_score=0

		random.shuffle(deck)

		player_cards=""
		player_cards=""

		player_cards+=str(deck.pop(0))
		player_cards+=str(deck.pop(0))

		