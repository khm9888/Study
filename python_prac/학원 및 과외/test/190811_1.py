#time() 지금 흐르고 있는 시간을 초로 표현해주는 함수
#목적? 현재 시간과 분.
'''
import time

x=time.time()

y=24*60*60 #오늘의 시간과 분과 초를 곱하면 하루의 초가 계산됨.
print(y)
z=x%y # 하루 이상의 값은 사라짐.

h = int(z/(60*60))
m = int((z%3600)/60)

print("오늘의 시간은 %d시 %d분입니다"%(h,m))
'''
'''
x="맑음"
y= 21
print("오늘의 날씨는 ",x,"입니다.온도는 ",y,"입니다.")
print("오늘의 날씨는 "+x+"입니다.온도는 "+str(y)+"입니다.")#형변환
print("오늘의 날씨는 %s입니다. 온도는 %d입니다."%(x,y))#포맷팅
# float-%f bool-%b str-%s int-%d
'''
y= ["a","b","c"]

z= []
for i in range(3):
	x=input("값넣어줘")
	z.append(x)
	print(z[::-1])
