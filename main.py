import sys
sys.path.append("/media/xavier/Samsumg/codes/ochlv_learner")
from experiment.exp2 import exp_config, preprocess, train, BinaryTargetsSet

# preprocess(exp_config)
# BinaryTargetsSet.prepare_full_data()
train(exp_config)
