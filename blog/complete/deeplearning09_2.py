#part9. RNN 데이터 자르기

import numpy as np

dataset=np.array(range(1,11))#(10,)

# 2.다대다

#array에서 앞에서부터 4개(처음엔 1~4) 값을 가져와서 그 다음값 2개(5,6)를 y로 설정하게
#split을 한다.

def split_2(dataset,time_steps,y_column):
    xs=list()
    ys=list()
    for i in range(0,len(dataset)-time_steps-y_column+1):#y의 개수가 늘어날수록 반복횟수가 줄어든다.
        x=dataset[i:i+time_steps]
        y=dataset[i+time_steps:i+time_steps+y_column]#y의 개수만큼 범위지정
        xs.append(x)
        ys.append(y)
    return np.array(xs),np.array(ys)

xs,ys=split_2(dataset,4,2)

print(f"xs:{xs}\nys:{ys}")

#입력(many) : 출력(one)

# #리스트 10개의 값을 넣고, 4개씩 자르고, 출력은 2열로
# def split_3(dataset,time_steps,y_column):
#     xs=list()
#     ys=list()
#     for i in range(0,len(dataset)-time_steps-y_column+1):
#         x=dataset[i:i+time_steps,:-1]
#         y=dataset[i+time_steps:i+time_steps+y_column,-1]
#         # print(f"{x}:{y}")
#         xs.append(x)
#         ys.append(y)
#     return np.array(xs),np.array(ys)
# # print(x,y=split(dataset,4))
# print(x)
# print("result")
# x,y=split_3(dataset,3,2)

# print(f"{x.shape}")
# print(f"{y.shape}")
# print(f"{x}\n\n{y}")

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