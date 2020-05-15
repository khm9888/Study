from numpy as np

x=np.Array(range(1,101))
y=np.Array(range(1,101))

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=66,test_size=0.4,shuffle=False)

x_train,x_valid,y_train,y_valid=train_test_split(x_train,y_train,)

from sklearn.model_selection import traion_test