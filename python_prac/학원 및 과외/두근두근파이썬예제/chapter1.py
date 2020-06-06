#p50

#8번

#t.up(),t.down,t.goto()등의 함수를 활용하여
#두 선을 그려라.

#터틀모듈가져오기
#터틀 선언
#200픽셀 전진
#터틀 up
#0,200으로 이동
#터틀 down
#200픽셀전진

import turtle

t=turtle.Turtle()
t.fd(200)
t.up()
t.goto(0,200)
t.down()
t.goto(200,200)



#9번
'''
터틀그래픽으로 오륜기를 그려라.
첫번째 원을 그린다.
그리고 두번째원의 시작 위치로 이동한다.
원을 그린다.
그리고 세번째원의 시작 위치로 이동한다.

import turtle

t=turtle.Turtle()
n=100

t.circle(n)#원을 그린다.

t.up()#두번째 원 시작 위치로 이동한다.
t.goto(150,0)#t.fd(150)
t.down()

#t.goto(0,150) #t.lt(90) t.fd(150)
t.circle(n)#원을 그린다.

t.up()#펜을 올려서 선이 안 그려지게 한다.
t.fd(150)#세번째 원 시작 위치로 이동한다.
t.down()#펜을 내려서 선이 그려지게 한다.

t.circle(n)#원을 그린다.

t.up()#네번째 원 시작 위치로 이동한다.
t.goto(75,-130)#t.fd(150)
t.down()

t.circle(n)#원을 그린다.

t.up()#펜을 올려서 선이 안 그려지게 한다.
t.fd(150)#다섯번째 원 시작 위치로 이동한다.
t.down()#펜을 내려서 선이 그려지게 한다.

t.circle(n)#원을 그린다.
'''
