# Training step #

1. Prepare the data:	data/list.txt,	data/label.txt,	data/features.txt,	data/pairs.txt
     
 >list.txt: go to image folder     
 ```
 printf '%s 0\n' $PWD/*/* > list.txt
 ```     
 >label.txt: put create_label.sh under image folder     
 ```
 ./create_label.sh > label.txt
 ```     
 >features.txt: follow caffe example of feature extraction    
 
 (Have to modify caffe/tools/extract_features.cpp first)
 ```
 cd caffe
 ./cmake_build/tools/extract_features models/vgg_face_caffe/VGG_FACE.caffemodel cmake_build/examples/_temp/VGG_FACE_deploy.prototxt fc7 cmake_build/examples/_temp/features 10 leveldb GPU 0 > features.txt
 ```     
 >pairs.txt :     
 ```
 ./generate_pair.py ../data/label.txt ../data/pairs.txt 500
 ```
2. Execute train

 >Modify test_lfw.py for data's path 
 
 >Execute test_lfw.py:
 ```
 ./test_lfw.py train
 ```

# Testing step #

  >Modify test_lfw.py for pairs.txt and features.txt
  
  >Adjust thresholds and positive_num and cv2.waitKey()
  ```
  ./test_lfw.py test
  ```     
