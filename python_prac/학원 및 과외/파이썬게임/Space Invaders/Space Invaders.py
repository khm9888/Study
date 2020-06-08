#Turtle Graphics Game
import turtle
import math
import random
import os

#Set up screen
wn = turtle.Screen()
wn.bgcolor("black")
wn.bgpic("kbgame-bg.gif")
#wn.tracer(30)

#Draw border
mypen = turtle.Turtle()
mypen.color("white")
mypen.up()
mypen.setposition(-300,-300)
mypen.down()
mypen.pensize(3)
for side in range(4):
	mypen.fd(600)
	mypen.lt(90)
mypen.hide()

#Create player turtle
player = turtle.Turtle()
player.color("blue")
player.shape("triangle")
player.up()
player.speed(0)

#Creat the score variable
score = 0

#Create goals
maxGoals = 10
goals = []

for count in range(maxGoals):
	goals.append(turtle.Turtle())
	goals[count].color("red")
	goals[count].shape("circle")
	goals[count].up()
	goals[count].speed(0)
	goals[count].setposition(random.randint(-300, 300), random.randint(-300, 300))

#Set speed variable
speed = 1

#Define functions
def turnleft():
	player.left(30)
	
def turnright():
	player.right(30)
	
def increasespeed():
	global speed
	speed += 1

def isCollision(t1, t2):
	d = math.sqrt(math.pow(t1.xcor()-t2.xcor(),2) + math.pow(t1.ycor()-t2.ycor(),2))
	if d < 20:
		return True
	else:
		return False	
#Set keyboard bindings
turtle.listen()
turtle.onkey(turnleft, "Left")
turtle.onkey(turnright, "Right")
turtle.onkey(increasespeed, "Up")

while True:
	player.forward(speed)
	
	#Boundary Checking
	if player.xcor() > 300 or player.xcor() < -300:
		player.right(180)
		#os.system("afplay bounce.mp3&")
		
	#Boundary Checking
	if player.ycor() > 300 or player.ycor() < -300:
		player.right(180)
		#os.system("afplay bounce.mp3&")
		
	#Move the goal
	for count in range(maxGoals):
		goals[count].forward(3)

		#Boundary Checking
		if goals[count].xcor() > 290 or goals[count].xcor() < -290:
			goals[count].right(180)
			#os.system("afplay bounce.mp3&")
				
		#Boundary Checking
		if goals[count].ycor() > 290 or goals[count].ycor() < -290:
			goals[count].right(180)
			#os.system("afplay bounce.mp3&")		

		#Collision checking
		if isCollision(player, goals[count]):
			goals[count].setposition(random.randint(-300, 300), random.randint(-300, 300))
			goals[count].right(random.randint(0,360))
			#os.system("afplay collision.mp3&")
			score += 1
			#Draw the score on the screen
			mypen.undo()
			mypen.penup()
			mypen.hideturtle()
			mypen.setposition(-290, 310)
			scorestring = "Score: %s" %score
			mypen.write(scorestring, False, align="left", font=("Arial",14, "normal"))




delay = raw_input("Press Enter to finish.")
