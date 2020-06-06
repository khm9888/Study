# f,s,t=map(int,input().split())#값을 입력
# values=[0]*f*s*t
# for i in range(1,f+1):
# 	for j in range(1,s+1):
# 		for k in range(1,t+1):
# 			sum=i+j+k
# 			values[sum]+=1
# print(values)
# print(values.index(max(values)))


# n,m = map(int,input().split())


# def method(m,i):


# for i in range(1,n+1):

# 	for j in range(m):
# 		print(i)


# def sol(n):
# 	for i in range(1,n+1):
# 		print(n-i+1)

# sol(4)

# def recursive(n,m=0):
# 	if m==0:
# 		print("어느 한 컴퓨터공학과 학생이 유명한 교수님을 찾아가 물었다.")
# 	print(m*"__",end="")
# 	print('"재귀함수가 뭔가요?"')
	
# 	if n!=0:
# 		print(m*"__",end="")
# 		print('"잘 들어보게. 옛날옛날 한 산 꼭대기에 이세상 모든 지식을 통달한 선인이 있었어.')
# 		print(m*"__",end="")
# 		print('마을 사람들은 모두 그 선인에게 수많은 질문을 했고, 모두 지혜롭게 대답해 주었지.')
# 		print(m*"__",end="")
# 		print('그의 답은 대부분 옳았다고 하네. 그런데 어느 날, 그 선인에게 한 선비가 찾아와서 물었어."')
# 		recursive(n-1,m+2)
# 	else:
# 		print(m*"__",end="")
# 		print('"재귀함수는 자기 자신을 호출하는 함수라네"')
# 	print(m*"__",end="")
# 	print("라고 답변하였지.")

# recursive(int(input()))

# n=int(input())

# for i in range(1,n+1):
# 	print("*"*(n+1-i))


# 4,2 -> 1~4까지 2개씩 뽑아서 수열
# n,m =map(int,input().split())
# check=[False]*(n+1)
# answer=[0]*m

# values=list()

# def recursive(n,m,index=0):
# 	if index==m:
# 		value=list()
# 		for i in range(m):
# 			value.append(answer[i])
# 		value.sort()

# 		if value not in values:
# 			for i in range(len(value)):
# 				if i==len(value)-1:
# 					print(answer[i])
# 				else:
# 					print(value[i], end=" ")
# 			values.append(value)
# 		return

# 	for i in range(1,n+1):
# 		if check[i]:
# 			continue
# 		check[i]=True
# 		answer[index]=i
# 		recursive(n,m,index+1)
# 		check[i]=False


# recursive(n,m)


# n=int(input())

# chess_table=[[True]*(n)]*(n)
# print(chess_table)
# cnt=0
# def recursive(n):
# 	for i in range(n):#행
# 		for j in range(n):#열
# 			if chess_table[i][j]:
# 				chess_table[i]=[False]*n
# 				for k in range(n):
# 					if chess_table[k][j]==True and 
# 					chess_table[k][j]=False

# 				print(chess_table)
# recursive(n)

# n=int(input())
# count=0
 
# row,left,right=[0 for _ in range(n)],[0 for _ in range(2*n-1)],[0 for _ in range(2*n-1)]#수직,왼쪽대각선,오른쪽 대각선
# #인덱스의 합과 차가 같은 대각선상에 있을때 같다는 것을 이용함
# #ex)0,2과 1,1과 2,0은 같은 대각선 상에 위치한다. 각행열의 합이 같은것을 알수있다.
 
# def queenlocation(index):
#     global count
#     if index==n:    #끝까지 퀸을 넣으면
#         count+=1
#         return
#     for col in range(n):  #열을 이동하며
#         if row[col] + left[index+col] + right[n-1+index-col]==0: #세조건에 걸리지 않는다면
#             row[col]=left[index+col]=right[n-1+index-col]=1
#             queenlocation(index+1)
#             row[col]= left[index+col]= right[n-1+index-col] = 0#초기화
 
# queenlocation(0)
# print(count)

# import itertools
# t=int(input())


# def combination(n,m):
# 	k=n-m
# 	answer=1
# 	while n>k:
# 		answer*=n
# 		n-=1
# 	while m>1:
# 		answer//=m
# 		m-=1

# 	return answer

# for _ in range(t):
# 	n,m=map(int,input().split())
# 	# result=list(itertools.combinations(range(m),n))
# 	result=combination(m,n)
# 	print(result)

# def factorial(n):
# 	if n!=1:
# 		return factorial(n-1)*n
# 	else:
# 		return 1
# print(factorial(3))

# n,m,v = map(int,input().split())

# graph=dict()
# for _ in range(m):
# 	i,j=map(int,input().split())
# 	if i not in graph.keys():
# 		graph[i]=set([j])
# 		# print(1)
# 	else:
# 		graph[i].add(j)
# 		# print(2)
# 	if j not in graph.keys():
# 		graph[j]=set([i])
# 		# print(3)
# 	else:
# 		graph[j].add(i)
# 		# print(4)

# def dfs(vertex,visited=list()):
# 	if vertex not in visited:
# 		visited.append(vertex)
# 		for n in graph[vertex].difference(visited):
# 			dfs(n,visited)
# 	return visited

# def bfs(vertex,visited=list()):
# 	queue=[vertex]
# 	visited.append(vertex)
# 	while queue:
# 		v=queue.pop(0)
# 		for n in graph[v].difference(visited):
# 			visited.append(n)
# 			#print("visited",visited)
# 			queue.append(n)
# 	return visited




# #print(graph)
# p= lambda a: (print(i,end=" ")for i in a)
# print(p(dfs(v)))
# print(bfs(v))