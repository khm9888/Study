'''
1~4사분면을 그리고, 원의 반지름이 1인 원을 그려보아라.

'''

import turtle as t1
import turtle as t2

t2.hideturtle()
t2.goto(100,0)
t2.up()
t2.goto(0,0)
t2.down()
t2.goto(-100,0)
t2.up()
t2.goto(0,0)
t2.down()
t2.goto(0,100)
t2.up()
t2.goto(0,0)
t2.down()
t2.goto(0,-100)

#t1.circle(100)
x=100
y=0
t1.up()
t1.goto(0,-100)
t1.down()
for i in range(400):
    if i<100:#제1사분면
        x-=1
        y=(100**2-x**2)**0.5
        t1.goto(x,y)
    elif i<200:#제2사분면
        x+=1
        y=(10000-x**2)**0.5
        t1.goto(-x,y)
    elif i<300:#제3사분면
        x-=1
        y=(10000-x**2)**0.5
        t1.goto(-x,-y)
    else:#제4사분면
        x+=1
        y=(10000-x**2)**0.5
        t1.goto(x,-y)

