def solution(s):
    answer = 0
    length=len(s)
    while len(s)>1:
        for l in range(length-1):
            if s[l]==s[l+1]:
                s=s[:l]+s[l+2:-1]
                
                answer=1
                break;


        print(s,answer)
            
    # [실행] 버튼을 누르면 출력 값을 볼 수 있습니다.
    print('Hello Python')

    return answer
solution("ac")