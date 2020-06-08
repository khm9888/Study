

# case_count=int(input())
# for _ in range(case_count):
# 	moving_distance=0
# 	s=0
# 	count=0

# 	start_point, end_point=map(int, input().split())

# 	total_distance=(end_point-1)-start_point
# 	remain_distance=total_distance
# 	while remain_distance>0:

# 		if s<=(total_distance//2):
# 			moving_distance+=1
# 		else:
# 			moving_distance-=1
# 		s+=moving_distance
# 		count+=1
# 		remain_distance=total_distance-s
# 		#print("remain_distance",remain_distance)
# 	count+=1
# 	#print("count",count)
# 	print(count)

# testCaseNum = int(input())
# for loopCount in range(testCaseNum):
#     testCase = list(map(int, input().split()))
#     distance = testCase[1] - testCase[0]

#     moveLength = 1
#     moveCount = 0

#     # 접근 방식 
#     # 어차피 처음도 끝도 결국에는 1로 이동해야되니까
#     # 그 이전 이동은 적어도 2여야한다. 그리고 그 이전은 또 3이어야하고
#     # 결국 1, 2, 3... 늘어났다가 다시 ...3, 2, 1로 줄어들어야하는 것.
#     # 총 거리에서 앞뒤로 이동할 수 있는 거리의 합을 계속 뺀다.
#     # 그러다가 남은거리가 가장 최근 이동한 거리의 합보다 같거나 작은 경우에는
#     # 현재의 moveLength와 같거나 작게 줄여나간다.
#     while distance > moveLength*2:
#         distance -= moveLength*2
#         moveLength += 1
#         moveCount += 2

#     while distance != 0:
#         if moveLength <= distance:
#             distance -= moveLength
#             moveCount += 1
#         else:
#             distance = 0
#             moveCount += 1

#     print(moveCount)

# case_count=int(input())
# numbers=list(map(int,input().split()))
# #print(numbers)
# count=0
# for i in range(case_count):
# 	check=True
# 	n=numbers[i]
# 	#print(n)
# 	if n!=1:
# 		for j in range(2,n):
# 			if n%j==0:
# 				check=False
# 	else:
# 		check=False
# 	if check:
# 		count+=1
# print(count)




# min=int(input())
# max=int(input())
# numbers=[]
# c_list=[]
# sum=0
# for i in range(min,max+1):
# 	numbers.append(i)

# #print(numbers)
# for i in range(max-min+1):
# 	check=True
# 	n=numbers[i]
# 	#print(n)
# 	if n!=1:
# 		for j in range(2,n):
# 			if n%j==0:
# 				check=False
# 	else:
# 		check=False
# 	if check:
# 		c_list.append(n)
# 		sum+=n
# if sum!=0:
# 	print(sum)
# else:
# 	c_list.append(-1)
# print(c_list[0])


# min,max=map(int,input().split())

# c_list=[]

# for i in range(min,max+1):
# 	c_list.append(i)

# for i in ()


# #print(numbers)
# for i in range(min,max+1):
# 	check=True
# 	n=i
# 	#print(n)
# 	if n!=1:
# 		for j in range(2,n):
# 			if n%j==0:
# 				check=False
# 	else:
# 		check=False
# 	if check:
# 		c_list.append(n)
		
# for i in c_list:
# 	print(i)

# a = input()
# num = list(map(int, a.split(' ')))

# N = num[1]
# num_range = [True] * (N+1)

# num_range[0] = False
# num_range[1] = False

# i = 2
# while i <= N**(1/2)+1:
#     if num_range[i] == True : #소수인 경우
#         j = i*i
#         while j <= N: #N까지
#             num_range[j] = False
#             j += i
#     i += 1

# M = num[0]
# for k in range(M, N+1):
#     if num_range[k]:
#         print(k)



# candi_numbers=[]

# while True:
# 	n=int(input())
# 	if n!=0:
# 		candi_numbers.append(n)
# 	else:
# 		break

# #print(candi_numbers)

# maximum=max(candi_numbers)*2

# num_range=[True]*(maximum+1)

# num_range[0]=False
# num_range[1]=False

# for i in range(2,maximum):
# 	if num_range[i]:
# 		j=i**2
# 		while j<=maximum:
# 			num_range[j]=False
# 			j+=i
# # for i in range(27):
# # 	print(i,num_range[i])

# for i in candi_numbers:
# 	print(num_range[i+1:i*2+1].count(True))


# x,y,w,h=map(int,input().split())

# print(min(x,y,(w-x),(h-y)))

# x_list=[]
# y_list=[]

# for _ in range(3):
# 	x,y=map(int,input().split())
# 	x_list.append(x)
# 	y_list.append(y)
# x_list.sort()
# y_list.sort()

# x_mid,y_mid=x_list[1],y_list[1]
# for _ in range(2):
# 	x_list.remove(x_mid)
# 	y_list.remove(y_mid)

# print(x_list[0],y_list[0])

# while True:
# 	length=list(map(int,input().split()))
# 	if 0 in length:
# 		break
# 	length.sort()

# 	if length[0]**2+length[1]**2==length[2]**2:
# 		print("right")
# 	else:
# 		print("wrong")

# import math

# r=int(input())

# print(math.pi*r**2)
# print((r**2)*2)
# case_count=int(input())
# for _ in range(case_count):
# 	x1,y1,r1,x2,y2,r2=list(map(int,input().split()))

# 	d=((x1-x2)**2+(y1-y2)**2)**0.5
# 	r_sum=r1+r2
# 	r_sub=abs(r1-r2)

# 	if d==0:
# 		if r_sub==0:
# 			print(-1)
# 		else:
# 			print(0)
# 	else:
# 		if d==r_sum or d==r_sub:
# 			print(1)
# 		elif  ((d<r_sum)and(d>r_sub)):
# 			print(2)
# 		else:
# 			print(0)
# stars=[]

# stars.append("***************************")
# stars.append("* ** ** ** ** ** ** ** ** *")
# stars.append("***************************")
# stars.append("***   ******   ******   ***")
# stars.append("* *   * ** *   * ** *   * *")
# stars.append("***   ******   ******   ***")
# stars.append("***************************")
# stars.append("* ** ** ** ** ** ** ** ** *")
# stars.append("***************************")
# stars.append("*********         *********")
# stars.append("* ** ** *         * ** ** *")
# stars.append("*********         *********")
# stars.append("***   ***         ***   ***")
# stars.append("* *   * *         * *   * *")
# stars.append("***   ***         ***   ***")
# stars.append("*********         *********")
# stars.append("* ** ** *         * ** ** *")
# stars.append("*********         *********")

# n=int(input())

# for i in range(1,n+1):
# 	print(stars[i%18-1])


# def stars(n):
#     matrix=[]
#     for i in range(3 * len(n)):
#         if i // len(n) == 1:
#             matrix.append(n[i % len(n)] + " " * len(n) + n[i % len(n)])
#         else:
#             matrix.append(n[i % len(n)] * 3)
#     return(list(matrix))

# star = ["***","* *","***"]
# n = int(input())
# k = 0
# while n != 3:
#     n = int(n / 3)
#     k += 1
    
# for i in range(k):
#     star = stars(star)
# for i in star:
#     print(i)

# count=[]
# def hanoi(n,a,b,c):
# 	if n==1:
# 		count.append((a,c))
# 	else:
# 		hanoi(n-1,a,c,b)
# 		count.append((a,c))
# 		hanoi(n-1,b,a,c)
# 	return count


# n=int(input())
# list=hanoi(n,1,2,3)
# print(len(list))
# for i in list:
# 	print(i[0],i[1])

# a,b=map(int,input().split())

# v=list(map(int,input().split()))
# m=0

# for i in range(len(v)):
# 	for j in range(i+1,len(v)):
# 		for k in range(j+1,len(v)):
# 			sum=v[i]+v[j]+v[k]
# 			if sum<=b and m<sum:
# 				m=sum
# print(m)

# n=int(input())
# check=False


# # 생성자일 수 있는 최소 범위값 정의(빠른 처리 속도를 위함)
# predictMin = n - 9*len(str(n))
# # 음수일 경우 0을 최소 범위값으로 재정의
# predictMin = 0 if predictMin < 0 else predictMin



# for number in range(predictMin, n):
# 	# number의 분해합 정의
# 	parsingSum = number
# 	parsingSum += sum(list(map(int, str(number))))
# 	#print("sum"+str(sum))	
# 	if parsingSum == n: 
# 		check=True
# 		print(number)
# 		break

# if not check:
# 	print(0)


# # 자연수 N 입력
# N = int(input())

# # 생성자일 수 있는 최소 범위값 정의(빠른 처리 속도를 위함)
# predictMin = N - 9*len(str(N))
# # 음수일 경우 0을 최소 범위값으로 재정의
# predictMin = 0 if predictMin < 0 else predictMin

# # 최소 범위값부터 N-1까지 반복하여 생성자 찾기
# for number in range(predictMin, N):
#   # number의 분해합 정의
#   parsingSum = number
#   parsingSum += sum(list(map(int, str(number))))

#   # 분해합이 N과 같으면 number가 생성자임
#   if parsingSum == N: break

# # 반복문을 지나고 number가 N-1과 같다면 생성자가 없는 것이므로 0을 반환하고, 있다면 number 반환
# print(0 if number == N-1 else number)

# case_count=int(input())
# casis=[]
# grade=[]
# for i in range(case_count):
# 	n=list(map(int, input().split()))
# 	casis.append(n)
# 	grade.append(1)
# 	# if i ==0:
# 	# 	max=n
# 	# elif max[0]<n[0] and max[1]<n[1]:
# 	# 	max=n

# #print(casis)

# for n in casis:
# 	for i,m in enumerate(casis):
# 		if m[0]<n[0] and m[1]<n[1]:
# 			grade[i]+=1
# for g in grade:
# 	print(g,end=" ")

a_count,b_count=map(int,input().split())
colors=[]

case1=[[0]*b_count for a in range(a_count)]
case2=[[0]*b_count for a in range(a_count)]



#print(a_count,b_count)
for a in range(a_count):
	text=input()
	row=[]
	for b in range(b_count):
		row.append(text[b])
		if (a+b)%2==0:
			case1[a][b]="W"
			case2[a][b]="B"
		else:
			case1[a][b]="B"
			case2[a][b]="W"
	colors.append(row)

c1=0
c2=0


for a in range(a_count):
	for b in range(b_count):
		if colors[a][b]!=case1[a][b]:
			c1+=1
		if colors[a][b]!=case2[a][b]:
			c2+=1
# c2=(a_count*b_count)-c1
for c in case1:
	print (c)
print(min(c1,c2))
