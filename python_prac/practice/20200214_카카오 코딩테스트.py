def solution(s):
	minlength=1000
	for c in range(1,(len(s)+1)//2+1):
		length=""
		count=1
		u=len(s)+c
		for x in range(0,u,c):
			if x ==0:
				pass
				#print(1)
			elif len(s[x:x+c])>c:
				pass
				#print(2)
			elif s[x-c:x]==s[x:x+c]:
				count+=1
				#print(3)
			else:
				#print(5)
				if count!=1:
					#print(str(count)+s[x-c:x],end="")
					length+=(str(count)+s[x-c:x])
				else:
					#print(s[x-c:x],end="")
					length+=s[x-c:x]
				count=1
		if minlength>len(length):
			minlength=len(length)
		#print()
	return minlength

# def solution(s):
# 	results=[]
# 	length=len(s)
# 	for c in range(1,int(len(s))+1):
# 		cnt=1
# 		before=""
# 		result=""
# 		for x in range(0,len(s)+1,c):
# 			try:
# 				now=s[x:x+c]
# 			except:
# 				now=s[x:]
# 			if now==before:
# 				cnt+=1
# 			elif x!=0:
# 				if cnt==1:
# 					cnt=""
# 				result=result+str(cnt)+before
# 				cnt=1
# 			before=now
# 		if c>len(now) and len(now)!=0:
# 			result+=now
# 		results.append(result)	
# 	for l in results:
# 		if length>len(l):
# 			length=len(l)
# 	return length
	
s="aabbaccc"
s1="ababcdcdababcdcd"
s2="abcabcdede"
s3="abcabcabcabcdededededede"
s4="xababcdcdababcdcd"
print(solution(s))
print(solution(s1))
print(solution(s2))
print(solution(s3))
print(solution(s4))
