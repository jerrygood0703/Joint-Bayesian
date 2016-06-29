# User Guide #
Training step:
>1. Prepare the data: data/list.txt, data/label.txt, data/features.txt, data/pairs.txt
     
 list.txt: go to image folder     
     ```bash
     printf '%s 0\n' $PWD/*/* > list.txt
     ```     
     label.txt: put create_label.sh under image folder     
     ```bash
     ./create_label.sh
     ```     
     features.txt: follow caffe example of feature extraction     
     ```bash
     cd caffe
     ./cmake_build/tools/extract_features models/vgg_face_caffe/VGG_FACE.caffemodel cmake_build/examples/_temp/VGG_FACE_deploy.prototxt fc7 cmake_build/examples/_temp/features 10 leveldb GPU 0 > features.txt
     ```     
     pairs.txt :     
     ```bash
     ./generate_pair.py ../data/label.txt
     ```
     
