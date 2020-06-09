import seaborn as sns
from matplotlib import pyplot as plt

df = sns.load_dataset('iris')

sns.regplot(x=df["sepal_length"], y=df["petal_length"])
