#p231
'''
import random

x=random.choice("abcde")
print(x)
'''

#p233
'''
# 거북이 경주게임

import turtle as t
import random as r
import time


def turtle_go(x,x1):
    s = 0
    s2 = 0
    while (s < 500)|(s2<500):
        y = r.randrange(100)
        x.fd(y)
        s += y

        y2 = r.randrange(100)
        x1.fd(y2)
        s2 += y2

t3 = t.Turtle()
t3.up()
t3.goto(350,-50)
t3.down()

t3.goto(350,250)

t1 = t.Turtle()
t1.shape("turtle")
t1.color("blue")
t1.up()
t1.goto(-200,0)
t1.down()

t2 = t.Turtle()
t2.shape("turtle")
t2.color("red")
t2.up()
t2.goto(-200,200)
t2.down()

time.sleep(2)
turtle_go(t1,t2)
'''

#p238
'''
import turtle
import random

p=turtle.Turtle()
p.up()
p.speed(0)
#s=p.getscreen()

a1=turtle.Turtle()
a1.shape("circle")
a1.up()
a1.speed(0)
a1.goto(random.randint(-300,300),random.randint(-300,300))

a2=turtle.Turtle()
a2.shape("circle")
a2.up()
a2.speed(0)
a2.goto(random.randint(-300,300),random.randint(-300,300))

'''
#p243

'''
import turtle
import math

player = turtle.Turtle()
player.shape("turtle")
screen = player.getscreen()

def turnleft():
    player.left(5)

def turnright():
    player.right(5)

def fire():
    x = 0
    y = 0
    velocity = 50
    angle = player.heading()
    vx = velocity*math.cos(angle*3.14/180)
    vy = velocity*math.cos(angle*3.14/180)
    while player.ycor()>=0:
        vx=vx
        vy = vy -10
        x = x + vx
        y = y + vy
        player.goto(x,y)

screen.onkey(turnleft, "Left")
screen.onkey(turnright, "Right")
screen.onkey(fire,"space")

screen.listen()
'''



