#20200215
#1번

# print("hello world!")

#2번


# print("강한친구 대한육군")
# print("강한친구 대한육군")

#3번

# print("\\    /\\")
# print(" )  ( ')")
# print("(  /  )")
# print(" \\(__)|")

#4번

# print("|\\_/|")
# print("|q p|")
# print('( 0 )"""\\')
# print('|"^"`    |')
# print('||_/=\\\\__|')

#5번

# a,b=3,4

# print(a+b)


#1000번, 두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

# a=int(input())
# b=int(input())

# b_1st=b%10
# b_2nd=(b-b_1st)%100
# b_3rd=b-b_2nd-b_1st

# # print(b_1st)
# # print(b_2nd)
# # print(b_3rd)

# value_3rd=a*b_1st
# value_4rd=a*b_2nd//10
# value_5rd=a*b_3rd//100
# value_6rd=a*b

# print(value_3rd)
# print(value_4rd)
# print(value_5rd)
# print(value_6rd)

#map 함수  - https://blog.naver.com/evey97/221696950510

# a,b = map(int, input().split())

# if a>b:
#     print(">")
# elif a<b:
#     print("<")
# else:
#     print("==")


# score = int(input())

# if score>=90:
# 	print("A")
# elif score>=80:
# 	print("B")
# elif score>=70:
# 	print("C")
# elif score>=60:
# 	print("D")
# else:
# 	print("F")

# year = int(input())

# if (year%4==0 and year%100!=0) or year%400==0:
# 	print("1")
# else:
# 	print("0")

# a,b  = map(int, input().split())

# if b<=45:
# 	b+=15
# 	a-=1
# else:
# 	b-=45

# if a>=24:
# 	a=0
# elif a==-1:
# 	a=23
# print(a,b)


# s_list=[]
# while True:
#     try:
#         a,b=map(int,input().split())
#         s_list.append(a+b)
#     except:
#         break
# while len(s_list)!=0:
# 	print(s_list.pop(0))


#while 3번째 문제


# 0보다 크거나 같고, 99보다 작거나 같은 정수가 주어질 때 다음과 같은 연산을 할 수 있다. 

# 먼저 주어진 수가 10보다 작다면 앞에 0을 붙여 두 자리 수로 만들고, 각 자리의 숫자를 더한다. 

# 그 다음, 주어진 수의 가장 오른쪽 자리 수와 앞에서 구한 합의 가장 오른쪽 자리 수를 이어 붙이면 새로운 수를 만들 수 있다. 다음 예를 보자.

# 26부터 시작한다. 2+6 = 8이다. 새로운 수는 68이다. 6+8 = 14이다. 새로운 수는 84이다. 8+4 = 12이다. 새로운 수는 42이다. 4+2 = 6이다. 새로운 수는 26이다.

# 위의 예는 4번만에 원래 수로 돌아올 수 있다. 따라서 26의 사이클의 길이는 4이다.

# N이 주어졌을 때, N의 사이클의 길이를 구하는 프로그램을 작성하시오.


# x=int(input())
# y=int(x)

# count=0
# while True:
# 	count+=1
# 	y1=y//10
# 	y2=y%10
# 	z=(y1+y2)%10
# 	last=y2*10+z
# 	y=last

# 	if last==x:
# 		break

# print(count)

# N =int(input())
# N2=N
# i=0
# while True:
#     T = N2//10
#     O = N2%10
#     TO = (T + O)%10
#     N2 = O*10 + TO
#     i+=1
#     if N2 == N:
#         break
# print(i)

#1차원 배열 1번


# n=int(input())
# values=input()
# n_list=values.split()
# min=10000000
# max=-10000000
# for i in n_list:
# 	v=int(i)
# 	if v<min:
# 		min=v
# 	if v>max:
# 		max=v
# print(min,max)

# v_list=[]
# max=-1
# for i in range(9):
# 	value=int(input())
# 	v_list.append(value)
# 	if max<value:
# 		max=value
# print(max)
# print(v_list.index(max)+1)

# a=int(input())
# b=int(input())
# c=int(input())

# r=str(a*b*c)
# c_list=[]
# for i in range(10):
# 	c_list.append(0)

# for i in r:
# 	c_list[int(i)]+=1

# for i in c_list:
# 	print(i)

# list_a=[]
# list_b=[]
# for i in range(10):
# 	list_a.append(int(input()))

# for i in list_a:
# 	list_b.append(i%42)

# list_b=list(set(list_b))
# print(len(list_b))

# n=int(input())
# m=input()
# n_list=m.split()
# for i,j in enumerate(n_list):
# 	n_list[i]=int(j)

# #print(n_list)

# max=-1

# for i in n_list:
# 	if max<i:
# 		max=i

# #print(max)

# total=0
# average=0

# for j in n_list:
# 	total+=j/max*100

# average=total/len(n_list)
# print(average)



# n=int(input())
# score=[]
# ox=[]
# for i in range(n):
# 	ox.append(input())


# for check in ox:
# 	s=0
# 	n=0
# 	for c in check:
# 		if c=="X":
# 			n=0
# 		else:
# 			n+=1
# 			s+=n
# 	score.append(s)

# for s in score:
# 	print(s)


# n=int(input())

# n_list=[]

# for i in range(n):
# 	n_list.append(input().split())


# for i in n_list:
# 	for j,k in enumerate(i):
# 		i[j]=int(k)

# #print(n_list)
# average=[]
# for i in n_list:
# 	sum=0
# 	for j in range(1,len(i)):
# 		sum+=i[j]
# 	average.append(sum/(len(i)-1))
	
# #print(average)

# for n,i in enumerate(n_list):
# 	count=0
# 	for j in range(1,len(i)):
# 		if i[j]>average[n]:
# 			count+=1
# 	print("%0.3f" %round(count/i[0]*100,3)+"%")


# import sys
# input = sys.stdin.readline

# test_case = int(input())

# for _ in range(test_case):
#     data = input().strip().split(' ')
#     scores = list(map(float, data[1:]))
#     average = sum(scores) / len(scores)

#     above = 0
#     for score in scores:
#         if score > average:
#             above += 1

#     print(f'{(above/len(scores))*100:.3f}%')

# import sys

# input = sys.stdin.readline

# test_case=int(input())

# for _ in range(test_case):
# 	data = input().split()
# 	scores = list(map(float,data[1:]))
# 	average=sum(scores)/len(scores)

# 	above = 0

# 	for score in scores:
# 		if score>average:
# 			above+=1
# 	print("%0.3f%%" %round(above/len(scores)*100,3))

#함수 1번

# Python 2, Python 3, PyPy, PyPy3: def solve(a: list) -> int
# a: 합을 구해야 하는 정수 n개가 저장되어 있는 리스트
# (0 ≤ a[i] ≤ 1,000,000, 1 ≤ n ≤ 3,000,000)
# 리턴값: a에 포함되어 있는 정수 n개의 합 (정수)



# def solve(a):
# 	sum=0
# 	for i in a:
# 		sum+=i
# 	return sum


#c=int(input("몇개나 할까요?"))
# def d(n):
# 	s=0
# 	t=n
# 	while t>=10:
# 		s+=t%10
# 		#print(s)
# 		t=t//10
# 	s+=t
# 	#print(s)
# 	return n+s

# not_list=[]

# for i in range(1,10000+1):
# 	not_list.append(d(i))
	

# for i in range(1,10000+1):
# 	if i not in not_list:
# 		print(i)


# def han(n):
# 	#print("n",n)
# 	if n<10:
# 		return True
# 	elif n>=10:
# 		values=[]
# 		while n>=1:
# 			values.append(n%10)
# 			n=n//10
# 		#print(values)	
# 		check=[]
# 		for i in range(len(values)-1):
# 			check.append(values[i]-values[i+1])
# 		check=set(check)
# 		#print("check",check)
# 		if len(check)==1:
# 			return True
# 		else:
# 			return False

# n=int(input())
# count=0
# for i in range(1,n+1):
# 	if han(i):
# 		count+=1

# print(count)

#문자열 2번

# n=int(input())
# sentence=input()
# sum=0
# for s in sentence:
# 	sum+=int(s)

# print(sum)

# 문자열 3번

# word = input()
# words=[]
# for i in range(26):
# 	words.append(-1)

# for w in word:
# 	if words[ord(w)-97] == -1:
# 		words[ord(w)-97]=word.index(w)

# for w in words:
# 	print(w,end=" ")

# n=int(input())

# for i in range(n):
# 	m=input()
# 	count=int(m.split()[0])
# 	word=m.split()[1]
# 	for m in word:
# 		print(m*count,end="")
# 	print()

# import sys
# input  = sys.stdin.readline

# word=input().lower().rstrip()

# words=[]

# for i in range(26):
# 	words.append(0)

# for w in word:
# 	#print(ord(w)-97)
# 	words[ord(w)-97]+=1

# max=-1
# for w in words:
# 	if w>max:
# 		max=w

# count=0
# for w in words:
# 	if w>=max:
# 		count+=1

# if count >=2:
# 	print("?")
# else:
# 	print(chr(words.index(max)+97).upper())

apart={}


def people(floor,room):
    if (floor,room) in apart.keys():
        return apart[floor,room]
    else:    
        sum=0
        if floor==0:#0층의 경우
            sum=room
        else:#아닌경우
            for i in range(1,room+1):
                sum+=people(floor-1,i)
        apart[floor,room]=sum
        return sum

case_count=int(input())
for _ in range(case_count):
    floor=int(input())#2
    room=int(input())#3
    print(people(floor,room))
