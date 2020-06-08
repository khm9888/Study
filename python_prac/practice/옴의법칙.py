# 알고리즘 작성

# 사용자로부터 전류와 저항 값을 입력 받아서, V=IR
# 전압, 전력 값을 출력한다. W=VI

i=float(input("전류는?"))
r=float(input("저항은?"))

v=i*r
w=v*i
print(f'전류는 {i}이고, 저항은 {r}')
print(f'전압는 {round(v,2)}이고, 저항은 {round(w,2)}')
