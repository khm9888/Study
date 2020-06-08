#p224
'''
#2번

#육각형을 그리는 함수 하나를 만들고, 
#그 함수를 이용해서 벌집 모양을 만들어봐라. (두근두근 파이썬, p.224)

import turtle

t = turtle.Turtle()
t.shape("turtle")

def draw_hexagon ():    
    for i in range(6) :      
        t.up()
        t.fd(100)
        t.down()
        hexagon()
        t.rt(60)

def hexagon():
    for i in range(6) :
        if i !=0 :
            t.lt(60)
            t.fd(100)
        else:
            t.up()
            t.lt(60)
            t.fd(100)
            t.down()

draw_hexagon()
'''

#3번
'''
import turtle as t

def fun(x) :
    y=(x**2 + 1)*0.01
    return y

t.up()
t.goto(-100,-100)
t.down()
t.fd(150)

t.up()
t.goto(-100,-100)
t.setheading(90)
t.down()
t.fd(200)
t.up()
t.goto(-100,-100)
t.down()


for i in range(150):
    t.goto(i-100,fun(i)-100)
'''
