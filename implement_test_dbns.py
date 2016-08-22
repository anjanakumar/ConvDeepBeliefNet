import sys,time
sys.path.insert(0,'DeepLearningTutorials/code/')
sys.path.insert(0,'DeepLearningTutorials/data/')

import rbm,DBN
from rbm import test_rbm
from DBN import test_DBN

# A simple script to run some of the default tests to determine the speed of training different algorithms. I found that my result was slower for the RBM test (151.38 compared to 122.47), but seemed faster than the DBM test result quoted on deeplearning.net (On my ThinkPad T430 I get ~1.65 mins/epoch for the DBN compared to 2.2 mins/epoch there). Still waiting for the full result for the DBN though. 
if(sys.argv[1]=='rbm'): 
  test_rbm()
elif(sys.argv[1]=='dbn'):
  test_DBN()
