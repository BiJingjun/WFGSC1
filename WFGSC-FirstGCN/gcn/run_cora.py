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

#laList1 = [0.001, 0.01, 0.1, 0.5, 1, 3, 5, 7, 9, 10, 20, 30, 40, 50]

sname = ("cora1", "cora2", "cora3", "cora4", "cora5", "cora6", "cora7", "cora8", "cora9", "cora10" )


for dname in sname:

    opt['dataset'] = dname

    run(opt)

