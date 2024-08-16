import os
import numpy as np


# count = 0
# for root, dirs, files in os.walk('./'):
#     for filename in files:
#         if filename.endswith('.npy'):
#             filepath = os.path.join(root, filename)
#             print(filename, np.load(filepath).shape[0])
#             count += np.load(filepath).shape[0]
# print(count)
# exit()

temp = [1782, 1467 , 782 , 189 , 488 , 347,191 , 79 , 283,198,123 , 64,294,228 , 128 , 91,323,248 , 129 , 86,294 , 192 , 128]
count = 0
for i in range(len(temp)):
    count += temp[i]
    print(count)
    # exit()
