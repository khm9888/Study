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
e_start_h=""
e_start_m=""


#위의 변수에 대해서 값을 받는다.

max_f=int(input("최대층수를 입력해주세요(ex - 8) : "))
now_f=int(input("현재층수를 입력해주세요(ex - 4) : "))
move=input("방향(위/아래)를 입력해주세요(ex - 위) : ")
start_time=input("시작 시간을 입력해주세요(ex - 12:45) : ")
end_time=input("끝나는 시간을 입력해주세요(ex - 00:10) : ")
#start_time="12:45"
start_h=int(start_time[:2])
start_m=int(start_time[-2:])
#print(start_h)
#print(start_m)
end_h=int(end_time[:2])#end_time[0:2]
end_m=int(end_time[-2:])

print("시작시간 : %s, 현재층수 : %d, 방향 : %s 시작합니다."%(start_time,now_f,move))

while (start_h==end_h and start_m == end_m)==False:
        if start_h//10==0:
                e_start_h="0"+str(start_h)
        else:
                e_start_h=str(start_h)
        if start_m//10==0:
                e_start_m="0"+str(start_m)
        else:
                e_start_m=str(start_m)
                
        print("시간 - %s:%s 현재층 - %s 방향 : %s"%(e_start_h,e_start_m,now_f,move))
        start_m+=1
        if move=="위":
                now_f+=1
        elif move=="아래":
                now_f-=1
        if start_m==60:
                start_m=0
                start_h+=1
        if start_h==24:
                start_h=0
        if now_f==max_f:
                move="아래"
        elif now_f==min_f:
                move="위"





