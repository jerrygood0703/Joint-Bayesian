# Training step #
1. Prepare the data:	data/list.txt,	data/label.txt,	data/features.txt,	data/pairs.txt
     
 >list.txt: go to image folder     
     ```
     printf '%s 0\n' $PWD/*/* > list.txt
     ```     
 >label.txt: put create_label.sh under image folder     
     ```
     ./create_label.sh
     ```     
 >features.txt: follow caffe example of feature extraction     
     ```
     cd caffe
     ./cmake_build/tools/extract_features models/vgg_face_caffe/VGG_FACE.caffemodel cmake_build/examples/_temp/VGG_FACE_deploy.prototxt fc7 cmake_build/examples/_temp/features 10 leveldb GPU 0 > features.txt
     ```     
 >pairs.txt :     
     ```
     ./generate_pair.py ../data/label.txt
     ```
     
