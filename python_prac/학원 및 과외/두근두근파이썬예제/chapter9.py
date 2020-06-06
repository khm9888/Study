#276

#1번
'''
#사용자로부터 5개의 숫자를 읽어서 리스트에 저장하고 숫자들의 평균을 계산한다.

n=[]
s=0
for x in range(5):
        n.append(int(input("정수를 입력하세요")))

for x in range(5):
	s+=n[x]
print(type(len(n)))
print("평균은 %d" %(s/len(n)))
'''

#2번
'''
#주사위를 굴려서 1~6이 각각 몇 번 나오는지 계산을 해보자.

import random as r

dice = [0,0,0,0,0,0]

for x in range(1000):
        rand = r.randrange(0,6)
        dice[rand] +=1

for y in range(6):
        print("주사위가 %d인 경우는 %d"%(y+1, dice[y]))
'''

#3번

'''
#딕셔너리를 사용하여 친구들의 이름과 전화번호를 입력하라. 사용자로부터 친구들의 이름과 전화번호를 입력받고 저장한다.
#이름을 입력하지 않고, 엔터키를 누르면 검색모드가 된다.

#딕셔너리를 만든다.
#이름과 전화번호에 대한 변수를 만든다.
#while 문을 만들어서, 이름이 null 이라면 검색모드로 변환한다.
#이름으로 검색하여 전화번호가 출력되도록 한다.

dic  =  {}
name = "test"
number ="test"

while name!="":
    name=input("(입력모드)이름을 입력하시오:")
    if name !="":
        number=input("(입력모드)전화번호를 입력하시오:")
        dic[name]=number

name=input("(검색모드)이름을 입력하시오:")
result=dic[name]
print("%s의 전화번호는 %s입니다."%(name,result))

'''

#4번

'''
#색상을 리스트에 저장하고, 거북이의 색상으로 설정하고, 사각형을 그리는 프로그램을 만든다.

#터틀 모듈을 가져온다.
#터틀을 설정하고, 리스트를 생성한 후, 색(랜덤)을 넣는다.

#마우스 클릭한 곳에 사각형이 그려지게 한다.

import turtle as tu
import random as ran

t= tu.Turtle()
s= tu.Screen()

c=["red","yellow","blue","black"]
t.up()

def draw_square(x,y):
    t.goto(x,y)
    t.down()
    t.color(c[ran.randrange(4)])
    t.begin_fill()
    for x in range(4):
        t.fd(100)
        t.rt(90)
    t.end_fill()
    t.up()

s.onscreenclick(draw_square)
'''

#5번
'''
#색상을 리스트에 저장하고, 거북이의 색상으로 설정하고, 다각형(3~10)을 그리는 프로그램을 만든다.

#터틀 모듈을 가져온다.
#터틀을 설정하고, 리스트를 생성한 후, 색(랜덤)을 넣는다.

#마우스 클릭한 곳에 다각형이 그려지게 한다.
import turtle
import random

t= turtle.Turtle()
s= turtle.Screen()

c=["red","yellow","blue","black","green"]
t.up()

def draw(x,y):
    t.goto(x,y)
    t.down()
    t.color("black",c[random.randrange(5)])
    t.begin_fill()
    r=random.randrange(3,10)
    l=random.randrange(50,151,10)
    for x in range(r):
        t.fd(l)
        t.rt(360/r)
    t.end_fill()
    t.up()

s.onscreenclick(draw)
'''

#7번
'''
#인터넷 도메인의 약자와 해당되는 국가를 딕셔너리에 저장해보자. ex. key-"kr"/value-"대한민국"

domains={}

q=True

while q:
    n_domain=input("도메인 약자를 입력해주세요 :")
    nation =input("국가를 입력해주세요 :")
    domains[n_domain]=nation
    check = input("더 입력하시겠습니다?(그만 입력하시겠다면 enter 만 해주세요):")
    if check == "":
        q=False

for x in domains.keys():
    print("도메인은 %s 국가는 %s" %(x,domains[x]))
    #print("도메인은 "+x+" 국가는 "+domains(x)) 

#for x,y in domains.items():
#    print("도메인은 %s, 국가는 %s"%(x,y))

print(x in domains.keys())
'''

#8번
'''
#딕셔너리에 문제와 정답을 저장하고, 하나씩 꺼내서 사용자에게 제시하는 프로그램을 작성하자.
#정답과 문제에 대해서 입력받는다.

import random

dic ={}
answer="test"
question="test"
while answer!="":
    answer=input("(입력모드)답을 입력하시오_그만하고 싶다면 enter!:")
    if answer !="":
        question=input("(입력모드)문제를 입력하시오:")
        dic[answer]=question
#print(dic.keys())
#type() - 자료형을 알려주는 함수, 기본-int, float, str / 응용 -list, dic~~~
q=list(dic.values())
a=list(dic.keys())
x=random.randrange(len(dic))#0,1,2,3
u_a=input("문제 : %s? \n답:" %q[x])

if a[x]==u_a:
    print("right")
else:
    print("wrong")

'''


