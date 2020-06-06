#윤희원

import time

fsec = time.time()

d_sec = 60
d_min = 60
d_hour=24
d_year=365

year = 1970+fsec/(d_year*d_hour*d_min*d_sec)
print(year)
year_reminds=fsec%(d_year*d_hour*d_min*d_sec)
day = year_reminds/(d_hour*d_min*d_sec)
sec = fsec%d_sec

#12345초 
