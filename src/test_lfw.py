#!/usr/bin/env python
#coding=utf-8
import sys
import numpy as np
import cv2
from common import *
from scipy.io import loadmat
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from joint_bayesian import *
import matplotlib.pyplot as plt

def excute_train(train_data="../data/lfw_features.txt", train_label="../data/lfw_label.txt", 
				test_pair="../data/lfw_pair.txt", result_fold="../result/lfwpca=150/"):
    #data  = loadmat(train_data)['lbp_WDRef']
    #label = loadmat(train_label)['id_WDRef']
    
    # Load data and label
    data  = loaddata(train_data)
    label = loadlabel(train_label)
    print data.shape, label.shape
    # crop the extra repeat data
    if len(label) != len(data):
        diff = (len(label) - len(data))
        data = data[:diff] 
        print data.shape
    
    #Load testing pairs and exclude data from training set    
    pairlist = []
    with open(test_pair, 'r') as f:
        for line in f:
            sample = line.split('\t')
            for e in sample:
                pairlist.append(int(e) - 1)
    print "data loading..."
    label = np.delete(label, pairlist, axis=0)
    data = np.delete(data, pairlist, axis=0)
    print data.shape, label.shape
    #----------------------------------------------------
    # crop the person with only one image
    del_index = data_filter(data, label)
    label = np.delete(label, del_index, axis=0)
    data = np.delete(data, del_index, axis=0)
    print data.shape, label.shape
    print data
    # joint_bayesian.data_pre()
    # data predeal scale data to [-1,1]
    data = data_pre(data, result_fold)
    print data
    # joint_bayesian.PCA_Train()
    # pca training.
    pca = PCA_Train(data, result_fold, 150)
    data_pca = pca.transform(data)
    print data
    data_to_pkl(data_pca, result_fold+"pca_wdref.pkl")
    JointBayesian_Train(data_pca, label, result_fold)

def showphoto(first, second):
    img1 = cv2.resize(cv2.imread(first), (250,250), interpolation = cv2.INTER_CUBIC)
    img2 = cv2.resize(cv2.imread(second), (250,250), interpolation = cv2.INTER_CUBIC)
    concat_img = np.zeros((500, 250, 3), np.uint8)
    concat_img[:250, :] = img1
    concat_img[250:, :] = img2
    cv2.imshow("compare", concat_img)
    cv2.waitKey()
    
def excute_test(result_fold="../result/emp_pca=150/", test_pair="../data/emp_pair.txt", 
				test_data="../data/emp_test_features.txt", test_list="../data/emp_test_list.txt"):
    with open(result_fold+"A_con.pkl", "rb") as f:
        A = pickle.load(f)
    with open(result_fold+"G_con.pkl", "rb") as f:
        G = pickle.load(f)

    '''pair_list = loadmat(pairlist)['pairlist_lfw']
    test_Intra = pair_list['IntraPersonPair'][0][0] - 1
    test_Extra = pair_list['ExtraPersonPair'][0][0] - 1'''
    pairlist = []
    with open(test_pair, 'r') as f:
        for line in f:
            sample = line.split('\t')
            # the pair number starts from 1, but array index starts from 0
            pairlist.append([int(e)-1 for e in sample]) 
    photolist = []
    with open(test_list, 'r') as f:
        for line in f:
            photolist.append(line[:-3])
    #print test_Intra, test_Intra.shape
    #print test_Extra, test_Extra.shape

    #data  = loadmat(test_data)['lbp_lfw']
    data = loaddata(test_data)
    scaler = joblib.load(result_fold+"scale_model.m")
    data  = scaler.transform(data)
    print data
    clt_pca = joblib.load(result_fold+"pca_model.m")
    data = clt_pca.transform(data)
    print data
    #data_to_pkl(data, result_fold+"pca_lfw.pkl")
    #data = read_pkl(result_fold+"pca_lfw.pkl")
    
    # Parameters to change
    thresholds = [ -600 ]
    positive_num = 358
    #---------------------
    distances_p = []
    distances_n = []
    maximum = -100000
    minimum = 1000000
    total_it = 0
    for threshold in thresholds:
        f_count = 0
        t_count = 0
        start = time.time()
        for p in pairlist[:positive_num]:
            distance = Verify(A, G, data[p[0]], data[p[1]])
            distances_p.append(distance)
            if distance < minimum:
                minimum = distance
            #print distance, p[0], p[1] 
            if distance > threshold:
                t_count += 1
            #    print 'correct'
            else:
                print distance, p[0], p[1]  
            total_it += 1          
            #showphoto(photolist[p[0]], photolist[p[1]])
        distances_p = np.array(distances_p)
        median_p = np.median(distances_p)
        std_p = np.std(distances_p)
        for p in pairlist[positive_num:]:
           distance = Verify(A, G, data[p[0]], data[p[1]])
           distances_n.append(distance)
           if distance > maximum:
               maximum = distance
           #print distance, p[0], p[1]
           if distance <= threshold:
                f_count += 1
           #     print 'correct'
           #else:
           #     print 'wrong!'
           total_it += 1           
           #showphoto(photolist[p[0]], photolist[p[1]]) 
        end = time.time()
        print t_count, f_count, threshold        
        print 'Average time per Verify: ' + str((end-start) / total_it)
        distances_n = np.array(distances_n)
        median_n = np.median(distances_n)
        std_n = np.std(distances_n)
    print 'min: ' + str(minimum) + ' max: ' + str(maximum)
    print 'Positive median: ' + str(median_p) + ' / Negative median: ' + str(median_n)
    print 'Positive std: ' + str(std_p) + ' / Negative std: ' + str(std_n) 
    #distances_p = np.sort(distances_p)
    #distances_n = (np.sort(distances_n))[::-1]
    plt.figure(figsize=(20, 7))
    plt.plot(distances_p, color="blue")
    plt.plot(distances_n, color="red")  
    plt.axis([0.0, total_it-positive_num, 100.0, median_n])
    plt.savefig(result_fold + "plot.jpg")    
    plt.show()
    '''dist_Intra = get_ratios(A, G, test_Intra, data)
    dist_Extra = get_ratios(A, G, test_Extra, data)

    dist_all = dist_Intra + dist_Extra
    dist_all = np.asarray(dist_all)
    label    = np.append(np.repeat(1, len(dist_Intra)), np.repeat(0, len(dist_Extra)))

    data_to_pkl({"distance": dist_all, "label": label}, result_fold+"result.pkl")'''


if __name__ == "__main__":
    if len(sys.argv) < 2:
    	print 'Usage: ./test_lfw.py train/test'
    elif str(sys.argv[1]) == 'train':
        print 'Training...'
        excute_train()
    elif str(sys.argv[1]) == 'test':
        print 'Testing...'
        excute_test()
    else:
        print 'train or test ??'
    #excute_performance("../result/result.pkl", -16.9, -16.6, 0.01)
