import time

x=time.time() #초단위로 현재 시간을 반영한다. flaot

s=int(x)%60
m=int(x%(60*60))//60
h=int(x%(60*60*24))//(60*60)

print(h,m,s)
