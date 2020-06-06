'''
print("hey \"hi\" hey")
u=input("원하는 값을 입력하세요")
x="It's sunny day!"
y=[10,"a",11.5]

print(x[0:1+1])

print(u[0]+u[1]+u[-2]+u[-1])
print(u[0:2]+u[-2:])
print(u[:2]+u[-2:])
'''
'''
y= ["a","b","c"]

z= []
for i in range(3):
	x=input("값넣어줘")
	z.append(x)
	print(z[::-1])
'''

import turtle
t=turtle.Turtle()

x=input("칼라를 고르시오")
y=input("2번쨰 칼라를 고르시오")
z=input("3번쨰 칼라를 고르시오") 
list=[x,y,z]

ch=int(input("세가지 값 중 어떤 값을 고르시겠습니다.(0,1,2 중에서 선택해주세요)"))
t.fillcolor(list[ch])
t.begin_fill()
t.circle(50)
t.end_fill()

print("true")
