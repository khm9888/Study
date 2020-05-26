# import matplotlib.pyplot as plt
# import cv2
# import numpy as np
# import time

# def aidemy_imshow(name,img):
#     b,g,r=cv2.split(img)
#     img=cv2.merge([r,g,b])
#     plt.imshow(img)
#     plt.show()

# cv2.imshow = aidemy_imshow


#p454

import numpy as np
import cv2

img=cv2.imread("../source/cleansing_data/sample.jpg")
# img=cv2.imread("D://source/cleansing_data/sample.jpg")
cv2.imshow("sample",img)
