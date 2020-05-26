 #3장
 
#  #p50

# from matplotlib import pyplot as plt

# years=list(range(1950,2020,10))
# gdp=[300.2,543.3,1075.9,2862.5,5979.6,10289.7,14958.3]

# plt.plot(years,gdp,color="green",marker="o",linestyle="solid")

# plt.title("Nominal GDP")

# plt.ylabel("Billions of $")
# plt.show()

# #p51
# #막대그래프

# movies=[]
# num_oscars=[]
# movies.append("annie")
# num_oscars.append(5)
# movies.append("ben")
# num_oscars.append(11)
# movies.append("casa")
# num_oscars.append(3)
# movies.append("gandhi")
# num_oscars.append(8)
# movies.append("west")
# num_oscars.append(10)

# plt.bar(range(len(movies)),num_oscars)

# plt.title("My Favorite Movies")
# plt.ylabel("# of Academy Awards")

# plt.xticks(range(len(movies)),movies)

# plt.show()

# #p52

# from matplotlib import pyplot as plt

# from collections import Counter

# grades = [85,95,91,87,70,0,85,82,100,67,73,77,0]


# histogram = Counter(min(grade//10*10,90) for grade in grades)

# plt.bar([x+5 for x in histogram],histogram.values(),10,edgecolor=(0,0,0))

# plt.axis=([-5,105,0,5])

# plt.xticks([10*i for i in range(11)])
# plt.xlabel("Decile")
# plt.ylabel("# of Students")
# plt.title("Distribution of Exam 1 Grades")

# plt.show()

# #p53

# from matplotlib import pyplot as plt


# mentions=[500,505]
# years = [2017,2018]

# plt.bar(years,mentions,0.8)
# plt.xticks(years)
# plt.ylabel("# of times I heard someone say 'data science'")

# plt.axis([2016.5,2018.5,499,506])
# plt.title("Look at the 'Huge Icrease!!")
# plt.show()


# plt.axis([2016.5,2018.5,0,506])
# plt.title("Look at the 'Huge Icrease!!")
# plt.show()