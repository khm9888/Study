# 50

from matplotlib import pyplot as plt

years=list(range(1950,2020,10))
gdp=[300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]

plt.plot(years,gdp,color="green",marker="o",linestyle="solid")

plt.title("Nominal GDP")

plt.ylabel("Billions of $")
plt.show()

#p51
#막대그래프

movies=[]
num_oscars=[]
movies.append("annie")
num_oscars.append(5)
movies.append("ben")
num_oscars.append(11)
movies.append("casa")
num_oscars.append(3)
movies.append("gandhi")
num_oscars.append(8)
movies.append("west")
num_oscars.append(10)

plt.bar(range(len(movies)),num_oscars)

plt.title("My Favorite Movies")
plt.ylabel("# of Academy Awards")

plt.xticks(range(len(movies)),movie)

plt.show()