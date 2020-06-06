
import time


n_time = time.time()

n_sec = 60
n_minute=60
n_hour = 24
n_year=365
sec = int(n_time%n_sec)
minute = int((n_time%(n_minute*n_sec))/n_sec)
hour = int(n_time%(n_sec*n_minute*n_hour)/(n_minute*n_sec))
day = int(n_time%(n_sec*n_minute*n_hour)/(n_hour*n_minute*n_sec))
print("날짜 :",hour,"시간 :",minute,"초 :",sec)
