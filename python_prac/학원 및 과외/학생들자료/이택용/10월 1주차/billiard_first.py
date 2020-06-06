#입력값: 당구테이블의 크기(y,x)
y=int(input())#6
x=int(input())#8

move_x=0
move_y=0
dir_x="right"
dir_y="up"

#끝나는 조건
#print("test")
while True:
	#print("test")
	if dir_y=="up":
		move_y+=1
	if dir_y=="down":
		move_y-=1
	if dir_x=="right":
		move_x+=1
	if dir_x=="left":
		move_x-=1

	if move_x==x and move_y==0:
		print("bottom right pocket (%d, %d)"%(move_x,move_y))
		break
	elif move_x==0 and move_y==y:
		print("top left pocket (%d, %d)"%(move_x,move_y))
		break
	elif move_x==x and move_y==y:
		print("top right pocket (%d, %d)"%(move_x,move_y))
		break
        #종료조건
             

#이동방향
	#print("x,y: (%d,%d)"%(move_x,move_y))
	if dir_y=="up" and move_y==y:
		print("top cushion (%d, %d)"%(move_x,move_y))
	elif dir_y=="down" and move_y==0:
		print("bottom cushion (%d, %d)"%(move_x,move_y))
	if dir_x=="right" and move_x==x :
		print("right cushion (%d, %d)"%(move_x,move_y))
	elif dir_x=="left" and move_x==0:
		print("left cushion (%d, %d)"%(move_x,move_y))
	#프린트, 쿠션에 닿는 경우

	#현재위치
	if move_y==y:
		dir_y="down"
	elif move_y==0:
		dir_y="up"
	if move_x==x:
		dir_x="left"
	elif move_x==0:
		dir_x="right"
	#print("x,y의 방향: %s,%s " %(dir_x,dir_y))
	#input()
	#wait(1000)





