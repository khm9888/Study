#클릭하는 곳에 사각형 그리기

import turtle as t

def squ (x):
    for i in range(4):
        t.fd(x)
        t.lt(90)

def draw (x,y):
    t.up()
    t.goto(x,y)
    t.down()
    t.begin_fill()
    t.color("green")
    squ(50)
    t.end_fill()

s = t.Screen()
s.onscreenclick(draw)

#s.listen() -? 확인필요.
