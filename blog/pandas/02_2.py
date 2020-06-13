import pandas as pd

data = {"fruits": ["apple", "orange", "banana", "strawberry", "kiwifruit"],
        "year": [2001, 2002, 2001, 2008, 2006],
        "time": [1, 4, 5, 6, 3]}
df = pd.DataFrame(data)
print(df)
#        fruits  year  time
# 0       apple  2001     1
# 1      orange  2002     4
# 2      banana  2001     5
# 3  strawberry  2008     6
# 4   kiwifruit  2006     3

print(df.index % 2 == 0)
print()
print(df[df.index % 2 == 0])

# [ True False  True False  True]

#       fruits  year  time
# 0      apple  2001     1
# 2     banana  2001     5
# 4  kiwifruit  2006     3