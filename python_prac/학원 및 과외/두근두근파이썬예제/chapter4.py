#p105

#값을 3번 받아서, 도형을 그린다.
#1. 도형모양, 2. 가로길이. 3.세로길이.

'''
import turtle

t=turtle.Turtle()

x=turtle.textinput("","")
y=turtle.textinput("","")

iy=int(y)

if x == "s":
    for i in range(0.1,0.4,0.1):
        t.fd(iy)
        t.lt(360/4)
elif x == "t":
    for i in range(3):
        t.fd(iy)
        t.lt(360/3)
elif x == "c":

    t.circle(iy)
y1=2020
m1=9
d1=1

m=int(input())
'''

#p108
'''
import turtle

t=turtle.Turtle()

s=turtle.textinput("","이름은?")

t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)
t.fd(100)
t.rt(90)

t.rt(90)
t.write("반갑습니다 "+s+"님")
'''


