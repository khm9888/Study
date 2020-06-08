#p71

#goto로 사각형 그리기 문제 
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

#p78
'''
#사용자로부터 몇 각형을 그릴지 입력받고, 그 값을 토대로 도형그리기

#터틀모듈 추가
#사용자로부터 값 받고, 변수에 넣기
#터틀 움직임을 명령한다. 단 입력받은 값에 따라 이동하는 각도 조절한다.

import turtle
t=turtle.Turtle()

angle=int(input("몇각형을 원하세요?:"))

for x im range(angle):
    t.fd(100)
    t.lt(360/angle)
'''
#p82

#투입금액, 물건값이 있는 자동판매기에서 물건 구입 후,
#거스름돈을 500원, 100원짜리로 받는 문제
#500원과 100원의 개수!!

#투입금액, 물건값을 input()함수로 받는다.
#거스름돈은 계산하여 그 금액으로 500원과 100원을 계산한다..
#500원은 거스름돈을 500원으로 나눠 떨어지는 몫.
#100원은 거스름돈을 500으로 나누어 남는 나머지.

a=int(input("얼마 넣으실래요?"))#4000
b=int(input("물건값은 얼마인가요?"))#2100

c=a-b#1900

coin500=c//500#3
coin100=(c%500)//100#4
print("거스름돈은"+str(c)+"입니다.")

print("500원은 "+str(coin500)+"개입니다.")
print("100원은 "+str(coin100)+"개입니다.")
'''


#goto로 사각형 그리기 문제 
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

#p92
#3번

'''
#사용자로부터 정수 4자리를 입력받아서, 각 자리수의
#합을 구하시오.
#사용자로부터 input()을 사용해서 숫자를 받는다.
#1000의 자리수를 구한다.
#100의 자리수
#10의 자리수
#1의 자리수
#각 자리수의 합을 구한다.
#출력을 한다.

a=int(input("4자리 숫자를 입력하세요:"))
#1234
s=a//1000#1
s_r=a%1000#234
h=s_r//100#2
h_r=a%100#34
t=h_r//10#3
o=h_r%10#4
total=s+h+t+o
print(total)
'''

#4번

'''
#사용자로부터 두 좌표를 입력받아서 그 사이 거리를 구하라.

#첫 번째 좌표값을 입력받는다.
#두 번째 좌표값을 입력받는다.
#두 개의 좌표에 대한 계산을 한다.
#결과를 출력한다.

x1,y1,x2,y2=int(input("x1을 입력하세요.")),int(input("y1을 입력하세요.")),int(input("x2을 입력하세요.")),int(input("y2을 입력하세요."))
result=((x1-x2)**2+(y1-y2)**2)**0.5

print("결과는 %d입니다."%result)
'''

#7번
'''
#time 함수를 사용해서 현재의 시간과 분을 구하라.


#time 모듈을 가져온다.
#현재의 초를 구한다.(time 함수)
#현재의 초를 사용해서 당일의 분을 구한다.
#현재의 초를 사용해서 당일의 초를 구한다.
#결과를 출력한다.

import time

t = time.time()#현재의 초

print(t)#4232323242

sec=t%60 #초 0~59
#60*60*24
#1m = 60s, 1h = 60m, 1day = 24h
min=t%(60*60)//60
print(min)
print()#~분 ~초
day=60*60*24

h=t%day
'''
