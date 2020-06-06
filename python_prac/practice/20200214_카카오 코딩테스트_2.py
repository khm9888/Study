# def solution(p):
# 	u=""
# 	v=""
# 	for i,check in enumerate(p):
# 		count1=0
# 		count2=0
		
		
# 		if check=="(":
# 			count1+=1
# 		elif check==")":
# 			count2+=1

# 		if count2>count1:
# 			u=solution(p[:i])
# 			v=solution(p[i:])

# 			result=""
# 			for word in w:
# 				if word=="(":
# 					result+=")"
# 				else:
# 					result+="("
# 			p="("+result+")"




# 	p=u+v
# 	return p


def check(p):
    l = 0
    for i in p:
        if i == '(':
            l += 1
        else:
            l -= 1
            if l < 0:
                return False
    return l == 0


def seperate(p):
    l = r = 0
    for i in range(len(p)):
        if p[i] == '(':
            l += 1
        else:
            r += 1
        if l > 0 and l == r:
            return p[:i+1], p[i+1:]


def solution(p):
    if p == '':
        return p
    u, v = seperate(p)
    if check(u):
        return u + solution(v)
    return f"({solution(v)}){''.join([i == ')' and '(' or ')' for i in u[1:-1]])}"

	# answer =''


	# return answer
s1="(()())()"
s2=")("
s3="()))((()"
#print(solution(s1))
print(solution(s2))
#print(solution(s3))
