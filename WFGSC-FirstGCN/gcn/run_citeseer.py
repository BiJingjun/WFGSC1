

import sys
import os
import copy
import json
import datetime

opt = dict()

#opt['dataset'] = "citeseer"
#opt['weight_decay'] = 5e-2

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))


#laList1 = [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 9, 10, 20, 30, 40, 50]


sname = ("citeseer1", "citeseer2", "citeseer3", "citeseer4", "citeseer5", "citeseer6", "citeseer7", "citeseer8", "citeseer9", "citeseer10" )


for dname in sname:

    opt['dataset'] = dname

    run(opt)