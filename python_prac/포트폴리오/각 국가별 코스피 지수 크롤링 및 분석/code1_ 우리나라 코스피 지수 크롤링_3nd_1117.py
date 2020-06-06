import pandas as pd
from urllib.request import urlopen
import bs4
import json
import datetime
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import math
import numpy as np
from sklearn.linear_model import LinearRegression
historical_prices = dict()
import tensorflow.compat.v1 as tf
import sys
#print(sys.path)


last_page=8 # 몇페이지까지 데이터를 가져올 것인가?

# 파이썬에서의 날짜 형식으로 바꿔주기

def date_format(d):

    d = str(d).replace('-', '.')
    yyyy = int(d.split('.')[0])
    mm = int(d.split('.')[1])
    dd = int(d.split('.')[2])
    this_date = datetime.date(yyyy, mm,dd)
    return this_date

def domesitc(index_cd ="KPI200", page_n=1, last_page=last_page):

	naver_index = "https://finance.naver.com/sise/sise_index_day.nhn?code="+index_cd+"&page="+str(page_n)
	source = urlopen(naver_index).read()
	source = bs4.BeautifulSoup(source, "html.parser")
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

    # 다음페이지 호출
	if page_n < int(last_page*10/6+1):
		page_n = page_n + 1
		domesitc(index_cd, page_n, last_page) 

	return historical_prices

def read_json(symbol,page,dic=dict()):

	url="https://finance.naver.com/world/worldDayListJson.nhn?symbol="+symbol+"&fdtc=0&page="+str(page)
	raw = urlopen(url)
	data = json.load(raw)

	for n in range(len(data)):
		date =pd.to_datetime(data[n]["xymd"]).date()
		
		price = float(data[n]["clos"])
		dic[date] = price
	while page<last_page:# 개수 정해서 했습니다.
		if(len(data))>=9:
			
			page+=1
			read_json(symbol,page,dic=dic)

	return dic

def exchange_rate(nation ="USDKRW", page_n=1, last_page=last_page,dic=dict()):

	exchange_url = "https://finance.naver.com/marketindex/exchangeDailyQuote.nhn?marketindexCd=FX_"+nation+"&page="+str(page_n)
	source = urlopen(exchange_url).read()
	source = bs4.BeautifulSoup(source, "html.parser")
	dates = source.find_all('td', class_='date') # <td class="date"> 태그에서 날짜 수집
	prices = source.find_all('td', class_='num')# <td class="number_1> 태그에서 price 수집
	#print(prices)
	for n in range(len(dates)):

		if dates[n].text.split('.')[0].isdigit():
        # 날짜 처리
			this_date = dates[n].text
			this_date = date_format(this_date)
        # 환율 처리
		this_rate = prices[n*2].text
		this_rate = this_rate.replace(',', '')
		this_rate = float(this_rate)

        # 딕셔너리에 저장
		dic[this_date] = this_rate

	if last_page == 0:

        # 마지막 페이지 찾기
		last_page = source.find('td', class_='pgRR').find('a')['href']
		last_page = last_page.split('&')[1]
		last_page = last_page.split('=')[1]
		last_page = int(last_page)

    # 다음페이지 호출
	if page_n < last_page:
		page_n = page_n + 1
		exchange_rate(nation, page_n, last_page,dic) 

	return dic

#국내 코스피 지수

domestic_prices=domesitc()

# 해외지수
indices={
"SPI@SPX":'s&p 500'
,'SHS@000001':'shanghai'
,"NII@NI225":'nikkei 225'
}
rates={
"USDKRW":'us_rates'
,'JPYKRW':'jp_rates'
,"CNYKRW":'ch_rates'
}
# 환율

historical_indices = dict()
historical_rates = dict()

for key, values in indices.items():
	s = dict()
	print(key)
	s = read_json(key,1,s)
	print(key)
	historical_indices[values]=s

for key, values in rates.items():
	s = dict()
	print(key)
	s = exchange_rate(nation=key,dic=s)
	print(key)
	historical_rates[values]=s	

#tmp = {'kospi200':domestic_prices,'s&p 500':historical_indices['s&p 500'],'nikkei 225':historical_indices['nikkei 225']}
tmp = {'kospi200':domestic_prices,'s&p 500':historical_indices['s&p 500'],'nikkei 225':historical_indices['nikkei 225'],'shanghai':historical_indices['shanghai']
,'us_rates':historical_rates['us_rates'],'jp_rates':historical_rates['jp_rates'],'ch_rates':historical_rates['ch_rates']}

#print(domestic_prices)
price_df=pd.DataFrame(tmp)
price_df=price_df.ffill()
price_df=price_df.bfill()


delete_value=int(last_page*1.5)
print(price_df)
f = open("record.txt", 'w')
for x,y in tmp.items():
	f.write(str(x)+"\n")
	f.write(str(y)+"\n")
f.close()

plt.figure(figsize=(15,8))

median=int(last_page*7)

plt.plot((price_df[:-delete_value]['kospi200']/price_df['kospi200'].iloc[-median])*100, label="kospi200")

plt.plot((price_df[:-delete_value]['s&p 500']/price_df['s&p 500'].iloc[-median])*100,label="s&p 500")
plt.plot((price_df[:-delete_value]['nikkei 225']/price_df['nikkei 225'].iloc[-median])*100,label="nikkei 225")
plt.plot((price_df[:-delete_value]['shanghai']/price_df['shanghai'].iloc[-median])*100,label="shanghai")
plt.plot((price_df[:-delete_value]['us_rates']/price_df['us_rates'].iloc[-median])*100,label="us_rates")
plt.plot((price_df[:-delete_value]['jp_rates']/price_df['jp_rates'].iloc[-median])*100,label="jp_rates")
plt.plot((price_df[:-delete_value]['ch_rates']/price_df['ch_rates'].iloc[-median])*100,label="ch_rates")

plt.legend(loc=0)
plt.grid(True, color='0.7', linestyle=':', linewidth=1)

plt.show()


#두번째 수행과제 : 코스피지수 및 나머지 6개지수의 평균 분산 표준편차 중간값 최대값 최소값(요약통계)을 뽑음

results={'kospi200':price_df[:-delete_value]['kospi200'],'s&p 500':price_df[:-delete_value]['s&p 500']
,'nikkei 225':price_df[:-delete_value]['nikkei 225'],'shanghai':price_df[:-delete_value]['shanghai']
,'us_rates':price_df[:-delete_value]['us_rates'],'jp_rates':price_df[:-delete_value]['jp_rates']
,'ch_rates':price_df[:-delete_value]['ch_rates']}

def avg(list):
	sum=0
	for i in range(len(list)):
		sum+=list[i]

	avg=round(sum/len(list),2)
	return avg

def getMedian(a):
  a_len = len(a)                # 배열 요소들의 전체 개수 구하기
  if (a_len == 0): return None  # 빈 배열은 에러 반환
  a_center = int(a_len / 2)     # 요소 개수의 절반값 구하기

  if (a_len % 2 == 1):   # 요소 개수가 홀수면
    return a[a_center]   # 홀수 개수인 배열에서는 중간 요소를 그대로 반환
  else:
    return (a[a_center - 1] + a[a_center]) / 2.0

for key, value in results.items():
	total=0
	#print(type(list(value)))
	a=avg(list(value))
	vsum=0
	for i in value:
		total+=i
		vsum=vsum+(i-a)**2
	var=vsum/len(value)
	sqrt=math.sqrt(var)
	print("%s의 합 : %s 평균 : %s, 분산 : %s, 표준편차 : %s, 중간값 : %s, 최대값 : %s, 최소값 : %s"%(key,round(total,2),round(a,2),round(var,2),round(sqrt,2),getMedian(value),max(value),min(value)))

slopes=[]
intercepts=[]
############################################################################
y=results['kospi200']
x=results['s&p 500']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################


############################################################################
y=results['kospi200']
x=results['nikkei 225']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################

############################################################################
y=results['kospi200']
x=results['shanghai']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################

############################################################################
y=results['kospi200']
x=results['us_rates']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################

############################################################################
y=results['kospi200']
x=results['jp_rates']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################

############################################################################
y=results['kospi200']
x=results['ch_rates']

independent_var=np.array(x).reshape(-1,1)
dependent_var=np.array(y).reshape(-1,1)

regr=LinearRegression()
regr.fit(independent_var,dependent_var)

result={"Slope":regr.coef_[0,0],"Intercept":regr.intercept_[0],"R^2":regr.score(independent_var,dependent_var)}

print(result)
slopes.append(regr.coef_[0,0])
intercepts.append(regr.intercept_[0])
##################################################################################


plt.figure(figsize=(6,5))
plt.scatter(results['s&p 500'],results['kospi200'],marker=".")
plt.grid(True, color='0.7', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("s&p 500")

plt.show()

plt.figure(figsize=(6,5))
plt.scatter(results['nikkei 225'],results['kospi200'],marker=".")
plt.grid(True, color='0.6', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("nikkei 225")

plt.show()

plt.figure(figsize=(6,5))
plt.scatter(results['shanghai'],results['kospi200'],marker=".")
plt.grid(True, color='0.5', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("shanghai")

plt.show()

plt.figure(figsize=(6,5))
plt.scatter(results['us_rates'],results['kospi200'],marker=".")
plt.grid(True, color='0.4', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("us_rates")

plt.show()

plt.figure(figsize=(6,5))
plt.scatter(results['jp_rates'],results['kospi200'],marker=".")
plt.grid(True, color='0.3', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("jp_rates")

plt.show()

plt.figure(figsize=(6,5))
plt.scatter(results['ch_rates'],results['kospi200'],marker=".")
plt.grid(True, color='0.2', linestyle=':', linewidth=1)
plt.ylabel("kospi200")
plt.xlabel("ch_rates")

plt.show()


##################################################################################



number_of_points = 200
x_point = []
y_point = []
a = slopes[0]
b = intercepts[0]
for i in range(number_of_points):
	x = np.random.normal(0.0,0.5)
	y = a*x + b +np.random.normal(0.0,0.1)
	x_point.append([x])
	y_point.append([y])

plt.plot(x_point,y_point, 'o', label='Input Data')
plt.legend()
plt.show()

# A와 b를 tf.Variable로 정의하고 임의의 값을 할당
A = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# A는 -1에서 1사이의 임의의 값으로, b는 0으로 초기화
B = tf.Variable(tf.zeros([1]))
# y와 x의 선형 관계식 정의
y = A * x_point + B
# 비용함수(cost function) 정의: 예측값과 실제값의 차이 -> mean squared error (MSE)
cost_function = tf.reduce_mean(tf.square(y - y_point))
# tensorflow에서 경사하강법(gradient descent)을 이용하여 cost_function을 최소화
optimizer = tf.train.GradientDescentOptimizer(0.5) # 0.5는 학습률(learning rate)
train = optimizer.minimize(cost_function)
# 변수 초기화
model = tf.global_variables_initializer()
# A와 b의 값을 도출할 수 있게 세션을 통해 모델 학습을 20회 반복하도록 설정
with tf.Session() as session: # 모델 시뮬레이션을 수행
	session.run(model)
	for step in range(0,21):
		session.run(train) # 각 스텝마다 학습을 수행
		if (step % 5) == 0: # 매 5번째 스텝마다 점이 어떤 패턴인지 출력
			plt.plot(x_point, y_point, 'o',label='step = {}'.format(step))
			# 학습된 A와 b를 이용한 회귀직선 y=Ax+b 출력
			plt.plot(x_point, session.run(A) * x_point + session.run(B))
			plt.legend()
			plt.show()