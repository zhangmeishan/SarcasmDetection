cmake .. -DEIGEN3_INCLUDE_DIR=/c/eigen/ -DN3L_INCLUDE_DIR=/d/workspace/LibN3L-2.0/
cmake .. -DEIGEN3_INCLUDE_DIR=~/workspace/eigen/ -DN3L_INCLUDE_DIR=~/workspace/LibN3L-2.0/

#sparse
-l -train D:\data\sarcasm\sarcasm1v1.train1.nn -dev D:\data\sarcasm\sarcasm1v1.dev1.nn -test D:\data\sarcasm\sarcasm1v1.test1.nn -option D:\data\sarcasm\option.sparse
#NNWord
-l -train D:\data\sarcasm\sarcasm1v1.train1.nn -dev D:\data\sarcasm\sarcasm1v1.dev1.nn -test D:\data\sarcasm\sarcasm1v1.test1.nn -option D:\data\sarcasm\option.word

#sparse
./SparseDetector -l -train ../newcorpus/1v1/sarcasm1v1.train1.nn -dev ../newcorpus/1v1/sarcasm1v1.dev1.nn -test ../newcorpus/1v1/sarcasm1v1.test1.nn -option option.sparse >sparse.log &
./NNWordLocal -l -train ../newcorpus/1v1/sarcasm1v1.train1.nn -dev ../newcorpus/1v1/sarcasm1v1.dev1.nn -test ../newcorpus/1v1/sarcasm1v1.test1.nn -option option.word >word.log &