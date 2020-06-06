# # 1757 달려달려

# #n분, 지침지수, m보다 커지면 달릴 수 없다.
# #i분에 달렸다면 Di만큼 달릴 수 있다. 

# #또한 학생들이 쉬기 시작하면 지침지수가 0이 되기 전에는 달릴 수가 없다.


# n,m = map(int,input())
# i_list=[]
# for i in range(n):
# 	i_list.append(input())

# distance=0
# for i in range(n):#5번
# 	for j in range(n-(i+1)):
# 		distance=i_list[0]+

# #https://www.acmicpc.net/problem/1757

# 1720 타일 코드

# n=int(input())

# tile =[]*31
# value=0
# for i in range(1,31):
# 	if i ==1:
# 		tile[i]=1
# 	elif i==2:
# 		tile[i]=3


# n=int(input())
# value=[]

# v_list=[]
# for i in range(n):
# 	value.append(list())
# 	for j in range(i+1):
# 		value[i].append(0)
# 	v=list(map(int, input().split()))
# 	#print(v)
# 	v_list.append(v)


# for i in range(n):
# 	for j in range(i+1):
# 		if i==0:
# 			value[i][j]=v_list[i][j]
# 		else:
# 			if j==0:
# 				value[i][j]=v_list[i][j]+value[i-1][j]
# 			elif j==i:
# 				value[i][j]=v_list[i][j]+value[i-1][j-1]
# 			else:
#  				value[i][j]=max(v_list[i][j]+value[i-1][j],v_list[i][j]+value[i-1][j-1])


# print(max(value[n-1])
# import sys

# n,m = map(int,input().split())
# dic=dict()
# for i in range(1,n+1):
# 	dic[i]=input()

# #print(dic)
# for j in range(m):
# 	q== str(sys.stdin.readline()).strip()
# 	try:
# 		print(dic[int(q)])
# 	except Exception as e:
# 		print(list(dic.values()).index(q)+1)


		
import sys

N, M = map(int, input().split())
number_pokemon = 1
pokemon_dict1 = {}
pokemon_dict2 = {}

for _ in range(N):
    name = str(sys.stdin.readline()).strip()
    pokemon_dict1[number_pokemon] = name
    pokemon_dict2[name] = number_pokemon
    number_pokemon += 1

answer = []
for _ in range(M):
    pokemon = str(sys.stdin.readline()).strip()
    try:
        print(pokemon_dict1[int(pokemon)])
    except:
        print(pokemon_dict2[pokemon])