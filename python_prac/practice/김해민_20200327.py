#0327-1
#2차원 리스트 뒤집기 - ⭐️zip⭐️

# def solution(mylist):
# 	answer=list()
# 	for i in range(len(mylist)):
# 		answer.append(list())
# 	for n,i in enumerate(mylist):
# 		print("n",n)
# 		for m,j in enumerate(mylist[n]):
# 			print("m",m)
# 			answer[m].append(j)

# 	return answer


# mylist =[ [1,2,3], [4,5,6], [7,8,9] ]

# print(solution(mylist))

# 0327-2
# 모든 멤버의 type 변환하기

# def solution(mylist):
# 	answer = list(map(int,mylist))
# 	return answer

# mylist = ['1', '100', '33']

# for i in range(3):
# 	print(type(solution(mylist)[i]))

# 0327-3
# map 함수 응용하기

# def solution(mylist):
# 	answer = list(map(len, mylist))
# 	return answer

# 0327-4
# sequence 멤버를 하나로 이어붙이기

# def solution(mylist):
# 	answer = ''
# 	for i in mylist:
# 		answer+=i
# 	return answer

# print(solution(['1', '100', '33']))

# # join 사용 권장

# my_list = ['1', '100', '33']
# answer = ''.join(my_list)

# # 0327-5
# # 삼각형 별찍기

# # n = int(input().strip())
# # for i in range(n):
# # 	print('*'*(i+1))


# #list 에도 곱하기가 된다.
# n =2
# answer= [123, 456]*n

# print(answer)

# # 0327-6
# # 곱집합(Cartesian product) 구하기 - product

# # iterable1 = 'ABCD'
# # iterable2 = 'xy'
# # iterable3 = '1234'

# # for i in iterable1:
# #     for j in iterable2:
# #         for k in iterable3:
# #             print(i+j+k)

# #파이썬 스타일 - itertools.product

# import itertools

# iterable1 = 'ABCD'
# iterable2 = 'xy'
# iterable3 = '1234'
# itertools.product(iterable1, iterable2, iterable3)

# 0327-7
# 2차원 리스트를 1차원 리스트로 만들기

# def solution(mylist):
# 	answer = []
# 	for i in mylist:
# 		for j in i:
# 			answer.append(j)
# 	return answer

# print(solution([['A', 'B'], ['X', 'Y'], ['1']]))

# # 방법들

# my_list = [[1, 2], [3, 4], [5, 6]]
# answer = []
# for i in my_list:
#     answer += i
# 파이썬에서는
# 파이썬의 다양한 기능을 사용하면, for 문을 사용하지 않고도 리스트를 이어붙일 수 있습니다.

# my_list = [[1, 2], [3, 4], [5, 6]]

# # 방법 1 - sum 함수
# answer = sum(my_list, [])

# # 방법 2 - itertools.chain
# import itertools
# list(itertools.chain.from_iterable(my_list))

# # 방법 3 - itertools와 unpacking
# import itertools
# list(itertools.chain(*my_list))

# # 방법4 - list comprehension 이용
# [element for array in my_list for element in array]

# # 방법 5 - reduce 함수 이용1
# from functools import reduce
# list(reduce(lambda x, y: x+y, my_list))

# # 방법 6 - reduce 함수 이용2
# from functools import reduce
# import operator
# list(reduce(operator.add, my_list))

# # 방법 7 - numpy 라이브러리의 flatten 이용
# import numpy as np
# np.array(my_list).flatten().tolist()

# 0327-8
# 순열과 조합


# import itertools

# def solution(mylist):
# 	mylist.sort()
# 	answer = list(map(list,itertools.permutations(mylist)))
# 	return answer

# print(solution([2, 1]))

# 0327-9
# 가장 많이 등장하는 알파벳 찾기

# import string

# my_str = input().strip()
# d=dict()

# for i in string.ascii_lowercase:
#     d[i]=0

# for i in my_str:
# 	d[i]+=1

# max=0
# answer=""
# for i in d.keys():
# 	if d[i]==max:
# 		answer+=i
# 	elif d[i]>max:
# 		max=d[i]
# 		answer=i

# print(answer)

# # for 문과 if문을 한번에

# def solution(mylist):
# 	answer = []
# 	for i in mylist:
# 		if i%2==1:
# 			pass
# 		else:
# 			answer.append(i**2)
# 	return answer

# mylist = [3, 2, 6, 7]
# answer = [ i**2 for i in mylist if i %2 == 0]

# answer = [ i**2 for i in mylist if i%2==0]
# print(answer)

# 클래스의 인스턴스 출력하기 __str__

# class Coord(object):
#     def __init__ (self, x, y):
#         self.x, self.y = x, y
#     def __str__ (self):
#         return '[{}, {}]'.format(self.x, self.y)

# point = Coord(1, 2)

# print(point)

# #가장 큰 수, inf
# a=100
# b=[a<float('inf') for i in range(2) if a<float('inf')]

# print(b)
# print((lambda a,b:min(a,b))(100,float('inf')))

# from functools import reduce
# l=[1,2,3,4,5]
# a=reduce(lambda x,y:x+y,l)

# print(a)
# characters = []
# sentence = 'Be happy!'
# characters=[char for char in sentence]
		
# print(characters)
