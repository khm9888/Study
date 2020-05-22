#part9. RNN 데이터 자르기

import numpy as np

dataset=np.array([range(1,11),range(11,21),range(21,31)])
print(dataset.shape)#(3,10)

dataset=np.transpose(dataset)
print(dataset.shape)#(10,3)
print(dataset)#(3,10)

#5. 다입력, 다대다 모델(2)
def split_5(dataset,time_steps,y_column):
    xs,ys=list(),list()
    for i in range(len(dataset)-time_steps-y_column+1):
        x=dataset[i:i+time_steps,:]#0:3
        y=dataset[i+time_steps:i+time_steps+y_column,:]
        
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

xs,ys=split_5(dataset,3,2)

print(f"xs:{xs} \n\n\n ys:{ys}")