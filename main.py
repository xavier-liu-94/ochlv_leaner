import sys
sys.path.append("/media/xavier/Samsumg/codes/ochlv_learner")
from experiment.exp1 import exp_config, preprocess, train

# preprocess(exp_config)
train(exp_config)