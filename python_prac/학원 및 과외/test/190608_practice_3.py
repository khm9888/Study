import turtle

colors = ["red","purple","blue","green","yellow","orange"]

t = turtle.Turtle()

turtle.bgcolor("white")

t.speed(0)
t.width(2)
length = 10

s = turtle.textinput("", "이름을 입력하세요 : ")
t.write("안녕하세요?"+s+"씨, turtle 인사드립니다")

while length < 5:
    t.forward(length)
    t.pencolor(colors[length&5])
    t.right(89)
    length += 5

