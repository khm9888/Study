# n,m=map(int,input().split())

# numbers=list()

# def bfs(numbers,):
# 	if len(numbers)==m:
# 		for j,k in enumerate(numbers):
# 			if j!=len(numbers)-1:
# 				print(k,end=" ")
# 			else:
# 				print(numbers[-1])

# 	else:
# 		for i in range(1,n+1):
# 			numbers.append(i)
# 			bfs(numbers)
# 			numbers.pop()
# 			# print(i,end=" ")

# bfs(numbers)

# # for i in range(1,n+1):
# # 	for j in range(1,n+1):
# # 		print(i,j)

# nums=list(map(int,input().split()))
# check_num=0
# for n in nums:
# 	check_num+=n**2
# check_num=check_num%10
# print(check_num)
# import sys
# input=sys.stdin.readline
# hour,minute=map(int,input().split())

# if hour==0 and minute<45:
# 	hour,minute=23,minute+15
# elif minute>=45:
# 	minute-=45
# else:
# 	hour,minute=hour-1,minute+15

# print(hour,minute)


# scale=list(map(int,input().split()))
# ascending=0
# descending=0
# for i,j in enumerate(scale):
# 	if i!=0:
# 		if j>scale[i-1]:
# 			ascending+=1
# 		elif j<scale[i-1]:
# 			descending+=1
# if ascending==7:
# 	print("ascending")
# elif descending==7:
# 	print("descending")
# else:
# 	print("mixed")


# n,k=map(int,input().split())

# def factorial(n):
# 	if n==1 or n==0:
# 		return 1
# 	else:
# 		return factorial(n-1)*n

# answer=factorial(n)//(factorial(n-k)*factorial(k))
# print(answer)

#알고리즘

from random import randint

answer=randint(1,100)

guess=int(input("몇?"))

while True:
	if guess==answer:
		print(f'정답입니다. 정답은 {guess}')
		break
	elif guess<answer:
		print("up")
	else:
		print("down")
	guess=int(input("몇?"))




