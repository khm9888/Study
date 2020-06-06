
'''
x=100 #int()
y=20.5 #float()
z="float" #str()
u=False #bool()


q=" 10 "
int(q)
print("")
print()

x=int(input("원하는 값을 입력해보세요."))
'''

'''
원 그리기
-사용자에게 원의 반지름을 입력받아서 그 크기로 원을 그려라.
'''

'''
사용자한테 이름과 나이를 입력받는다. (input) 그에 따라 100살이 되는 연도를 계산하여 출력한다.
'''
'''
x= input("이름을 입력하세요")
y= int(input("나이를 입력해주세요."))

print(x+"씨"+str(2019-y+1+100)+"년도가 되면 100살이 되시네요")
print(x,y)
print(x+str(y))

'''
'''
x=list(range(10))
print(x)
str(int((3+4+5)/2))
'''
'''
import turtle
f= turtle.Turtle()
side=100

f.speed(1)

f.fd(side*2)
f.lt(90)
f.fd(side*2)
f.lt(90)
f.fd(side*2)
f.lt(90)
f.fd(side*2)
f.lt(90)

f.fd(side)
f.lt(90)
f.fd(side*2)
f.fd(-side)
f.rt(90)
f.fd(-side)
f.fd(side*2)
'''
'''
#goto로 사각형 그리기 문제 
f.goto(200,0)
f.goto(200,200)
f.goto(0,200)
f.goto(0,0)
f.goto(100,0)
f.goto(100,200)
f.goto(100,100)
f.goto(0,100)
f.goto(200,100)
'''
y=100
x=y

y=10

x,y=100,200 #x=100, y=200
