import sys,time
sys.path.insert(0,'DeepLearningTutorials/code/')
sys.path.insert(0,'DeepLearningTutorials/data/')
sys.path.insert(0,'lrn2/examples/mnist_pretrain/')
from run_demo import *

import rbm,DBN
from rbm import test_rbm
from DBN import test_DBN

''' A simple script to run some of the default tests to determine the speed of
 training different algorithms. 

I found that my result was slower for the RBM
 test (151.38 compared to 122.47), but seemed faster than the DBM test result
 quoted on deeplearning.net for LISA labs (On my ThinkPad T430 I get ~1.65 mins/epoch for the
 DBN compared to 2.2 mins/epoch there). The pretraining ran for 871.49 m and the fine tuning ran for 226.5 m, giving a best validation and test error of 1.49% (compared to 1.27 and 1.34 respectively for lisa labs) at iteration 110000. This is slower than the 615 minutes for pretraining and 101 minutes for fine tuning reported by lisa labs. Possibly the fact that I was running this and lrn2 at the same time may have slowed it down at the end. 

Started testing the default mnist_pretrain algorithm from lrn2 (the one with RBMs). Each epoch takes about 97 seconds. Full training took 590.5 m. 
'''
if(sys.argv[1]=='rbm'): 
  test_rbm()
elif(sys.argv[1]=='dbn'):
  test_DBN()
elif(sys.argv[3]=='mnist_pretrain'):
    parser = argparse.ArgumentParser(description = "Run a complete lrn2 work flow")

    parser.add_argument("run_keyword", metavar = "run_keyword", help = "Keyword for the current test")

    parser.add_argument("modelconfig", help = "model config file")
    parser.add_argument("nettype", help = "Type of net I am using")

    parser.add_argument("--re-train", action = "store_true", default = False,
                        help = "re-trains RBM, ignoring cached files corresponding to the run_keyword used")

    parser.add_argument("--load-existing", action = "store_true", default = False,
                       help = "loads existing parameters, trains non-existing")

    parser.add_argument("--rebuild-corpus", action = "store_true", default = False,
                        help = "reloads data and rebuilds temporary files for accelerating data import")

    args = parser.parse_args()
    config = get_config(args.modelconfig)

    test_mnist(args, config)

