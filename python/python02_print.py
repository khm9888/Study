# print 문과 format 함수
a = '사과'
b = '배'
c = '옥수수'

print('명석이는 잘생겼다.')

print(a)
print(a,b)
print(a,b,c)

print("나는 {0}를 먹었다.".format(a)) # why?  왜 이런 방식
print("나는 {0}와 {1}를 먹었다.".format(a,b)) # why?  왜 이런 방식
print("나는 {0}와 {1}와 {2}를 먹었다.".format(a,b,c)) # why?  왜 이런 방식



print('나는', a, '를  먹었다.')
print('나는',a, '와', b, '를 먹었다.') # why?  왜 이런 방식
print('나는' ,a, '와', b, '와', c, '를 먹었다.') # why?  왜 이런 방식

# 파라미터 sep = 
print('나는 ', a, '를  먹었다.', sep = '#')
print('나는 ',a, '와', b, '를 먹었다.', sep = '#')
print('나는 ' ,a, '와 ', b, '와 ', c, '를 먹었다.', sep = '#')

# +
print('나는', a+'를  먹었다.')
print('나는',a+'와', b+'를 먹었다.')
print('나는' ,a+'와', b+'와', c+'를 먹었다.')