

import sys
import os
import copy
import json
import datetime

opt = dict()


def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))

sname = ("largecora1", "largecora2", "largecora3", "largecora4", "largecora5",
         "largecora6", "largecora7", "largecora8", "largecora9", "largecora10")
#          "largecora11", "largecora12", "largecora13", "largecora14", "largecora15",
#          "largecora16", "largecora17", "largecora18", "largecora19", "largecora20")
#sname = ("largecora16", "largecora17", "largecora18", "largecora19", "largecora20")

#laList1 = [1.0e-4, 1.0e-3, 1.0e-2,0.1,1,10,100,300,600,800,1000,1500,2000,5000]
#laList1 = [1.0e-8, 1.0e-7, 1.0e-6,10000,20000,50000,100000,500000,1000000]
#laList1 = [1.0e-9, 1.0e-10, 1.0e-11,1000000,10000000,100000000,1000000000]
#laList1 = [1]

laList = [0, 0.1, 1, 500, 50000,100000,400000,3000000]

for dname in sname:
    for lamb in laList:
        opt['lamb'] = lamb

        opt['dataset'] = dname

        run(opt)