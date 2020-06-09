import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv("./dacon/data/train.csv",index_col=0,header=0,encoding="cp949")
x=train.iloc[:,:-4]
y=train.iloc[:,-4:]
# print(len(x))
# print(len(y.keys()))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

x_array = scaler.fit_transform(x)

# DataFrame 형태
# # print(x.iloc[:,0])
print(type(x_array))#np.array
print(x_array.shape)
print(x_array[0])

y_array = y.values

for i in range(len(x.keys())):
        fig=plt.figure()
        
        ax4=fig.add_subplot(221)
        txt=str(i)+" "+str(0)
        ax4.set_title(txt)
        ax4.scatter(x_array[:,i],y_array[:,0])
        
        ax1=fig.add_subplot(222)
        txt=str(i)+" "+str(1)
        ax1.set_title(txt)
        ax1.scatter(x_array[:,i],y_array[:,1])
        
        ax2=fig.add_subplot(223)
        txt=str(i)+" "+str(2)
        ax2.set_title(txt)
        ax2.scatter(x_array[:,i],y_array[:,2])
        
        ax3=fig.add_subplot(224)
        txt=str(i)+" "+str(3)
        ax3.set_title(txt)
        ax3.scatter(x_array[:,i],y_array[:,3])
        
        plt.show()
        

# for i in range(len(x.keys())):
#         fig=plt.figure()
        
#         ax4=fig.add_subplot(221)
#         txt=str(i)+" "+str(0)
#         ax4.set_title(txt)
#         ax4.hist(x_array[:,i])
        
#         ax1=fig.add_subplot(222)
#         txt=str(i)+" "+str(1)
#         ax1.set_title(txt)
#         ax1.hist(x_array[:,i])
        
#         ax2=fig.add_subplot(223)
#         txt=str(i)+" "+str(2)
#         ax2.set_title(txt)
#         ax2.hist(x_array[:,i])
        
#         ax3=fig.add_subplot(224)
#         txt=str(i)+" "+str(3)
#         ax3.set_title(txt)
#         ax3.hist(x_array[:,i])
        
#         plt.show()
        
        

# fig=plt.figure()

# ax4=fig.add_subplot(221)
# txt=str(0)
# ax4.set_title(txt)
# ax4.hist(y_array[:,0])

# ax1=fig.add_subplot(222)
# txt=str(1)
# ax1.set_title(txt)
# ax1.hist(y_array[:,1])

# ax2=fig.add_subplot(223)
# txt=str(2)
# ax2.set_title(txt)
# ax2.hist(y_array[:,2])

# ax3=fig.add_subplot(224)
# txt=str(3)
# ax3.set_title(txt)
# ax3.hist(y_array[:,3])

# plt.show()