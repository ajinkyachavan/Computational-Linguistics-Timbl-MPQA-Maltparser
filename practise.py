import numpy as np



with open('pos_tagged_train.txt') as f:
    mylist = [tuple(map(str, i.split(','))) for i in f]

print mylist[:10]
# f = open("train/pos_tagged_train.txt")
#
# k = 0
# for row in f.readlines():
#     print row
#     k += 1
#     if(k>10):
#         break
#
# k = 0
# for row in x:
#     print row
#     k += 1
#     if(k>10):
#         break