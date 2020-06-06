import turtle
import random

def draw_maze(x,y):
    for i in range(2):
        t.penup()
        if i == 1:
            t.goto(x+100,y+100)
        else :
            t.goto(x,y)
        t.pendown()
        t.forward(300)
        t.right(90)
        t.fd(300)
        t.lt(90)
        t.fd(300)

def turn_lt():
    t.left(10)
    t.forward(10)

def turn_rt():
    t.rt(10)
    t.fd(10)

t = turtle.Turtle()
screen = turtle.Screen()
t.shape("turtle")
t.speed(0)

x= 0
y= 0
x = int(input("시작위치 x값은?"))
y = int(input("시작위치 y값은?"))
draw_maze(x,y-50)

screen.onkey(turn_lt,"Left")
screen.onkey(turn_rt,"Right")

t.penup()
t.goto(x,y)
t.speed(1)
t.pendown()
screen.listen()
screen.mainloop()






