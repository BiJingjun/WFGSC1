

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





sname = ("photo1","photo2","photo3","photo4","photo5","photo6","photo7","photo8","photo9","photo10")

#laList = [0, 0.01, 0.1, 1, 10, 50, 100, 200, 500, 1000, 10000, 50000, 100000, 500000]
laList = [0,0.1,1,10,100,500,1000]

for dname in sname:
    for lamb in laList:
        opt['lamb'] = lamb

        opt['dataset'] = dname

        run(opt)