#p172
'''
for i in range(1000): #  숫자를 다룰 때 적합.
    print(i) #0~999
'''
x= "동해물과백두산~~~"
cnt=0

for c in x: #문자열, 리스트
    print(c) #6번의 횟수 반복문.
    cnt+=1
print("총글자는 %d" %cnt)

'''
x_2=["a","alpha",1]

for c in x_2: #리스트
    print(c) #3번의 횟수 반복문.
'''

#p183
'''
import random

com=random.randrange(1,101)
user=int(input("값을 맞춰주세요(1~100)"))
cnt=1

while user!=com:
    if com>user:
        print("값좀 올려요")
    else:
        print("값을 내려요")
    user=int(input("값을 맞춰주세요(1~100):"))
    cnt+=1
print("축하합니다. 시도횟수는 %d번 입니다."%cnt)
'''

#p195

#10번

'''
import turtle

tu=turtle.Turtle()

x=1
while True:
    tu.fd(100)
    tu.rt(90)
    tu.fd(15)
    tu.rt(90)
    tu.fd(100)
    tu.lt(90)
    tu.fd(15)
    tu.lt(90)
'''
