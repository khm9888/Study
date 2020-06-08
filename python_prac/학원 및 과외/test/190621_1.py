'''
import turtle as t

t.shape("turtle")

pos_colors = [[0,0,"Blue"],[-120,0,"Purple"],[60,60,"Red"],[-60,60,"Yellow"],[-180,60,"Green"]]

for x,y,z in pos_colors:
    t.up()
    t.goto(x,y)
    t.down
    t.color(z,z)
    t.begin_fill()
    t.circle(50)
    t.end_fill()
'''

x  = "python"
x = x[0:2]+x[-2:]
print(x)
