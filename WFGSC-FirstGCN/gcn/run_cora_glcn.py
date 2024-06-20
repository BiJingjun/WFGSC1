import sys
import os
import copy
import json
import datetime

opt = dict()

#opt['dataset'] = "cora"

def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))


laList = [0,0.01,0.1,1,10,50,100,500,1000]
#sname = ("cora1", "cora2", "cora3", "cora4", "cora5", "cora6", "cora7", "cora8", "cora9", "cora10" )


for lamb in laList:
    opt['lamb'] = lamb


    run(opt)

