import turtle as t


t.shape("turtle")

def draw_hexagon ():
    t.up()
    t.goto(0,0)
    t.fd(100)
    t.down()
    hexagon()
    
    for i in range(5) :      
        t.up()
        t.rt(60)
        t.fd(100)
        t.down()
        hexagon()

def hexagon():
    for i in range(6) :
        if i !=0 :
            t.lt(60)
            t.fd(100)
        else:
            t.up()
            t.lt(60)
            t.fd(100)
            t.down()


draw_hexagon()

    
    
