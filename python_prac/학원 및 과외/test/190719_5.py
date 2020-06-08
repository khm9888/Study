import turtle as t


t.shape("arrow")
t.speed(0)
t.seth(90)

def f(x):
    if x>20:
        t.fd(x)
        t.fd(-x/4)
        t.lt(60)
        f(x/4)
        t.rt(120)
        f(x/4)
        

for i in range(6):
    f(200)
    t.rt(60)
    t.up()
    t.goto(0,0)
    t.down()
