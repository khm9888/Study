#p135
#정수의 부호에 따른 거북이를 이동하시오.
'''
import turtle

t=turtle.Turtle()

s=int(input("정수를 입력해주세요."))
s=t.textinput("","정수를 입력해주세요.")

if s>0:
    t.goto(100,100)
    t.write("거북이가 이쪽으로 오면 양수입니다.")
elif s<0:
    t.goto(100,0)
    t.write("거북이가 이쪽으로 오면 0입니다.")
else:
    t.goto(100,-100)
    t.write("거북이가 이쪽으로 오면 음수입니다.")
'''
#p139
'''
import turtle

t=turtle.Turtle()

#입력을 받아서 l 이라고 하면 현재 각도에서 lt,
#r이라고 하면 현재 각도에 대해서 rt, set

x=input("l or r?(멈추고 싶다면 break)")#str
while True:
    if x=="l":
        t.lt(90)
        t.fd(100)
    elif x=="r":
        t.rt(90)
        t.fd(100)
    elif x=="break":
        break;
    else:
        print("잘못했네요 다시하세요.")

#동전뒤집기 앞/뒤
#가위바위보, 경우수 3
#주사위 굴리기 6
x=6
if x%2==0:
elif x%3==0:
else
'''
#p141
'''
#윤년판단

y=int(input("지금은 몇년인가요?"))

if (y%4==0 and y%100!=0) or y%400==0:
    print("윤년")
else:
    print("아닙니다.")
'''

#p148
'''
#로그인 
#아이디는 "tree", 비밀번호는 3434 로 설정해주고, 로그인 시도 프로그램을 만들어라.
#아이디, 비밀번호의 확인은 if 문을 사용해서 시도한다.  값의 입력은 input() 통해서 받는다.

i="tree"
pw=3434

i_u=input("아이디를 입력해주세요: ")
pw_u=input("비밀번호를 입력해주세요: ")
#elif
if i_u==i and pw==pw_u:
    print("로그인 성공하셨습니다.")
elif i_u != i:
    print("아이디를 틀렸습니다.")
elif pw_u != i:
    print("패스워드를 틀렸습니다.")
else:
    print("잘못된 경우입니다.")

#중복 if
if i_u==i:
    if pw_u==pw:
        print("로그인 성공하셨습니다.")
    else:
        print("비밀번호 오류")
else:
    print("아이디 오류")
'''
#p154

#7번
'''
#컴퓨터로 부터 2자리 숫자를 받는다(0~99), 그리고 사용자로부터도 2자리 숫자를 받아서, 십의 자리 숫자와 일의 자리 숫자가
#같은지 비교하여 2개가 모두 맞을 경우 100만원, 1개만 맞을 경우 50만원, 0개는 0원으로 출력한다.


import random as r

x=r.randint(0,99)#0~99  #56
y=int(input("0~99 맞춰봐요")) #65

c_1=x//10
c_2=x%10

u_1=y//10
u_2=y%10

#d1=c_1==u_1 #True or False

#if d1==True and d2==Ture

if c_1==u_1 or c_1==u_2:
    if c_2==u_1 or c_2==u_2:
        print(100)
    else:
        print(50)
else:
    if c_2==u_1 or c_2==u_2:
        print(50)

    else:
        print(50)
'''

