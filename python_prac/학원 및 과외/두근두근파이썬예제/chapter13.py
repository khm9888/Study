#p359
'''
from turtle import *
alex = Turtle()
alex.fd(100)
alex.lt(90)
alex.fd(100)

'''
#p360~362
'''
class Car:
    def drive(self):
        self.speed=10
        
myCar = Car()

myCar.color="red"
myCar.model="E-Class"
myCar.drive()#함수가 호출됨에 따라 값이 소환된다.
print(myCar.speed)
'''

#p363
'''
class Car:
    def __init__(self,speed,color,model):
        self.speed= speed
        self.color=color
        self.model=model

    def drive(self):
        self.speed=60

myCar= Car(0,"red","E-Class")

print(myCar.speed)
print(myCar.color)
print(myCar.model)
myCar.drive()
print(myCar.speed)
'''

#p364_다중객체 만들기-1개의 클래스로
'''
class Car:
    def __init__(self,speed,color,model):
        self.speed= speed
        self.color=color
        self.model=model

    def drive(self):
        self.speed=60

myCar=Car(0,"silver","E")
yourCar=Car(0,"red","S")
oldCar=Car(0,"orange","A")
'''

#p365 - __str()__메소드
'''
class Car:
    def __init__(self,speed,color,model):
        self.speed= speed
        self.color=color
        self.model=model

    def drive(self):
        self.speed=60

    def __str__(self):
        msg="속도:"+str(self.speed)+" 색상:"+self.color+" 모델:"+self.model
        return msg

myCar = Car()
print(myCar)
'''
#p367, 터틀그래픽
# 세개의 원을 그리는 그래픽 반복

import turtle

c=turtle.Turtle()
t=turtle.Turtle()
r=turtle.Turtle()
c.shape("circle")


