#part9. RNN 데이터 자르기

import numpy as np

dataset=np.array([range(1,11),range(11,21),range(21,31)])
# print(dataset.shape)#(3,10)

dataset=np.transpose(dataset)
# print(dataset.shape)#(10,3)
# print(dataset)


# 3.다대일, 다입력

# array에서 앞에서부터 3*2의 값을 가져와서 3번째 행의 가장 오른쪽 값(열:-1)을 가져온다.


def split_3(dataset,time_steps,y_column):
    xs=list()
    ys=list()
    for i in range(0,len(dataset)-time_steps+1):
        x=dataset[i:(i+time_steps),:-1]#
        y=dataset[i+time_steps-1:(i+time_steps-1)+y_column,-1]

        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

print("result")
xs,ys=split_3(dataset,3,1)

print(f"xs:{xs}\n\n ys:{ys}")

# #RNN split 다대다 모델
# def split_5(dataset,time_steps,y_column):
#     xs,ys=list(),list()
#     for i in range(len(dataset)-time_steps-y_column+1):
#         x=dataset[i:i+time_steps,:]#0:3
#         y=dataset[i+time_steps:i+time_steps+y_column,:]
        
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs), np.array(ys)

# x,y=split_5(dataset,3,2)

# print(f"{x} \n\n\n {y}")