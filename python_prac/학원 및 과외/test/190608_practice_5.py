

year = int(input("원하는 연도를 검색하세요"))
if ((year%4==0) & (year%100!=0) | (year%400==0) ):
	print("윤년입니다")
else :
    print("아니래요..")
