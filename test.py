<<<<<<< HEAD
persons=int(input())#몇명?
people = []#후보들 
for i in range(persons):
    people.append(int(input()))#숫자입력
    
dasom = people[0]
candidate = people[1:]
candidate = sorted(candidate)
cnt=0#몇번했는지 확인
if persons==1:
    pass
else:
    dasom = people[0]
    candidate = people[1:]
    candidate.sort()
    while dasom<=candidate[-1]:#무한루프
        candidate[-1]-=1
        dasom+=1
        candidate = sorted(candidate)
        cnt+=1      
        # print(people)
        # print(cnt,people)
print(cnt)
=======
#8393

#n이 주어졌을 때, 1부터 n까지 합을 구하는 프로그램을 작성하시오.

#알고리즘

#1. 사용자로부터 숫자를 입력받는다.
#2. 더해진 값을 받을 변수를 선언한다.
#3. 1부터 입력받은 값까지 반복문을 돌려서, 각 숫자의 합을 s에 더한다.
#4. s를 출력한다.

n= int(input())
s=0
for i in range(n):
    s+=i+1
print(s)

#1000

#두 정수 A와 B를 입력받은 다음, A+B를 출력하는 프로그램을 작성하시오.

#알고리즘

#1. 사용자로부터 숫자를 입력받는다. 단 1줄로 두 개의 값을 입력받는다.
#2. 두 값을 더해서 출력한다. 


a,b = map(int, input().split())
print(a+b)

#1271

#알고리즘

#1.사용자로부터 숫자를 입력받는다. 단 1줄로 두 개의 값을 입력받는다.
#2.두 값을 나눈 몫(정수)을 출력하고, 나머지도 출력한다.

n,m = map(int,input().split())
print(n//m)
print(n%m)
>>>>>>> cf8081287e242926645fa28799dd17cc40b6b563
