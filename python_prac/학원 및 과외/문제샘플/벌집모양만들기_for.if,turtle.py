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
