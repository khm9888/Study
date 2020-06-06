import turtle as t1
import random as r

t1.shape("turtle")
t1.color("red")

t1.up()
t1.goto(-300,0)
t1.down()

t2.shape("turtle")
t2.color("blue")

t2.up()
t2.goto(-300,200)
t2.down()

for i in range(100):
    t1.fd(r.randint(1,30))
    t2.fd(r.randint(1,30))
