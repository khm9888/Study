from urllib.request import urlopen
import bs4
import datetime
import json
import pandas as pd

# last_page=5 # 몇페이지까지 데이터를 가져올 것인가?
cnt=720
last_page_domestic=cnt//6 #5페이지
last_page_foreign=cnt//10 #3페이지

print("last_page_domestic")
print(last_page_domestic)
print("last_page_foreign")
print(last_page_foreign)
#코스피는 6개씩,
#S&p는 10개씩,
#공배수는 30개

historical_prices = dict()

def date_format(d):
	
	d = str(d).replace('-', '.')
	yyyy = int(d.split('.')[0])
	mm = int(d.split('.')[1])
	dd = int(d.split('.')[2])
	this_date = datetime.date(yyyy, mm,dd)
	return this_date

def domesitc(index_cd ="KPI200", page_n=1, last_page=last_page_domestic):#bs4 이용해서 가져옴

	naver_index = "https://finance.naver.com/sise/sise_index_day.nhn?code="+index_cd+"&page="+str(page_n)
 
	source = urlopen(naver_index).read() #from urllib.request import urlopen 통해서 가져오고
	source = bs4.BeautifulSoup(source, "html.parser") #Beautifulsoup 통해서 html parsing
 
	dates = source.find_all('td', class_='date') # <td class="date"> 태그에서 날짜 수집
	prices = source.find_all('td', class_='number_1') # <td class="number_1> 태그에서 price 수집
 
	for n in range(len(dates)):

		if dates[n].text.split('.')[0].isdigit():
		# 날짜 처리
			this_date = dates[n].text
			this_date = date_format(this_date)
		# 종가 처리
		this_close = prices[n*4].text
		this_close = this_close.replace(',', '')
		this_close = float(this_close)

		# 딕셔너리에 저장
		historical_prices[this_date] = this_close

	if last_page == 0:

		# 마지막 페이지 찾기

		last_page = source.find('td', class_='pgRR').find('a')['href']
		last_page = last_page.split('&')[1]
		last_page = last_page.split('=')[1]
		last_page = int(last_page)
	if page_n==1:
		print("-"*10,index_cd,"end","-"*10)

	print("kospi200",page_n)
	# 다음페이지 호출

	if page_n < int(last_page):
		page_n = page_n + 1
		domesitc(index_cd, page_n, last_page)
	elif page_n == int(last_page):
		print("-"*10,index_cd,"end","-"*10)

	return historical_prices

#국내 코스피 지수

domestic_prices=domesitc()

def foreign(symbol,page,dic=dict(),last_page=last_page_foreign):
	
	url="https://finance.naver.com/world/worldDayListJson.nhn?symbol="+symbol+"&fdtc=0&page="+str(page)
	raw = urlopen(url)
	data = json.load(raw)

	for n in range(len(data)):
		date =pd.to_datetime(data[n]["xymd"]).date()
		price = float(data[n]["clos"])
		dic[date] = price
	print(symbol,page)
	if page<last_page:# 개수 정해서 했습니다.
		page+=1
		foreign(symbol,page,dic=dic)
	 

	return dic

# 해외지수
indices={
"SPI@SPX":'snp500'
,'SHS@000001':'shanghai'
,"NII@NI225":'nikkei225'
}

historical_indices = dict()


for key, values in indices.items():
	s = dict()
	print("-"*10,key,"start","-"*10)
	s = foreign(key,1,s)
	print("-"*10,key,"end","-"*10)
	historical_indices[values]=s
 
 
# print("historical_indices")
# print(historical_indices)

print("len(domestic_prices)")
print(len(domestic_prices))
print("len(historical_indices['snp500'])")
print(len(historical_indices['snp500']))
print("len(historical_indices['nikkei225'])")
print(len(historical_indices['nikkei225']))
print("len(historical_indices['shanghai'])")
print(len(historical_indices['shanghai']))

tmp = {'kospi200':domestic_prices,'snp500':historical_indices['snp500'],'nikkei225':historical_indices['nikkei225'],'shanghai':historical_indices['shanghai']
}

#print(domestic_prices)
price_df=pd.DataFrame(tmp)
price_df.iloc[0,1:]=price_df.iloc[0,1:].fillna(1)

print(price_df.head())

# price_df=price_df.bfill()
price_df=price_df.dropna()

price_df = price_df.sort_index()

price_df.to_csv("./project/pro1/test.csv")
