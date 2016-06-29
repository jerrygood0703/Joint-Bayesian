#!/usr/bin/env python
#coding=utf-8
import sys
import numpy as np
import random

data = []
with open(str(sys.argv[1]), 'r') as f:
    for line in f:           
        sample_data = line.split('\t')
        data.append([int(e) for e in sample_data])
label = np.array(data)
#print label[206]
num = 1
pairlist = []
tlist = []
for count in range(label.shape[0]):
    l = label[count]
    if l == num:
        tlist.append(count)
    else:
        pairlist.append(tlist)
        num += 1
        tlist = []
        tlist.append(count)
pairlist.append(tlist)		
npp = np.array(pairlist)
print 'class number: ' + str(npp.shape)
#-----------
# for debug
#----------
for p in npp:
    if len(p) == 0:
        print 'fuckkkkkk'
    #print p
#-----------
rand = []
for p in pairlist:
    if len(p) >= 9: 
        rand.append(random.sample(range(len(p)), 4))
    else:
        rand.append([])

f = open('pairs.txt', 'w')
for x,y in zip(rand, pairlist):
    count = 0
    for i in range(len(x)):
        if count == 0:
            f.write(str(y[x[i]]+1)+'\t')
        else:
            f.write(str(y[x[i]]+1))
        count += 1
        if count == 2:
            f.write('\n')
            count = 0
#f.close()

diff = []
for i in range(len(pairlist)):
    num = random.sample(range(len(pairlist[i])), 1)[0]
    for j in xrange(i+1, len(pairlist)):
        pair = random.sample(range(len(pairlist[j])), 1)[0]
        diff.append([pairlist[i][num], pairlist[j][pair]])
diff_rand = random.sample(range(len(diff)), 250)
#f = open('pairs.txt', 'a')
for d in diff_rand:
    f.write(str(diff[d][0]+1)+'\t'+str(diff[d][1]+1)+'\n')
f.close()


