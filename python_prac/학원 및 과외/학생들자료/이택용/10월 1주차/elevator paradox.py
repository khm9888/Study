#1번 엘레베이터 문제
#엘레베이터의 최대 층수와, 현재 층수, 움직일 방향 그리고 시작한 시간, 종료 시간을 입력받아서 엘레베이터의 움직임을 출력하라.

#엘레베이터의 최대 층수와, 현재 층수, 움직일 방향 그리고 시작한 시간, 종료 시간을 변수선언 후 입력받는다.
#현재시간이 종료시간이 되면, 반복문을 멈춘다 (= 종료시간이 될 때까지 1분씩 더하면서 반복을 한다.)
#현재시간이 60분이 될 때 0으로 바꾼다.
#60분이 될 때 시간에 1을 플러스해준다.
#방향이 위라면, 현재층에서 1분이 지날 때마다,(반복문이 실행될 때마다) +1층을 해준다.
#방향이 아래라면, 현재층에서 1분이 지날 때마다,(반복문이 실행될 때마다) -1층을 해준다.
#최고층이 될 때, 방향을 아래로 전환/0층이 될 때 방향을 위로 전환
#언제까지? 종료시간 - while? 왜?


max_f=0#int
now_f=0
min_f=0
move=""#str
start_h=0
start_m=0
end_h=0
end_m=0

#위의 변수에 대해서 값을 받는다.

time=int(input("지속시간을 입력해주세요(ex - 10) : "))
now_f=int(input("현재층수를 입력해주세요(ex - 4) : "))
max_f=int(input("최대층수를 입력해주세요(ex - 8) : "))
move=input("방향(위/아래)를 입력해주세요(ex - ^/v) : ")#^,V로 표현변경
start_h=int(input("시작 시(h)을 입력해주세요(ex - 12) : "))
start_m=int(input("시작 분(m)을 입력해주세요(ex - 45) : "))
type_print=input("verbose/quiet")
floor=now_f

#start_time="12:45"

#print(start_h)
#print(start_m)
end_h=start_h
end_m=start_m+time
while end_m>=60:
    if end_m>=60:
        end_m-=60
    end_h+=1
    if end_h==24:
        end_h=0
    

print("시작시간 : %d:%d 끝나는 시간 : %d:%d 현재층수 : %d, 방향 : %s 시작합니다."%(start_h,start_m,end_h,end_m,now_f,move))

while (start_h==end_h and start_m == end_m and end_m<=60)==False:

	if type_print=="verbose":
		print("%d:%d %s [%s]"%(start_h,start_m,now_f,move))
	elif type_print=="quiet" and floor==now_f:
		print("%d:%d %s [%s]"%(start_h,start_m,now_f,move))
	start_m+=1
	if move=="^":
		now_f+=1
	elif move=="v":
		now_f-=1
	if start_m==60:
		start_m=0
		start_h+=1
	if start_h==24:
		start_h=0
	if now_f==max_f:
		move="v"
	elif now_f==min_f:
		move="^"




