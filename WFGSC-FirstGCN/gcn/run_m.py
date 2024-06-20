import sys
import os
import copy
import json
import datetime

opt = dict()

#opt['dataset'] = "m"

# opt['losslr1'] = 0.001
# opt['decay_lr'] = 1.0
# opt['lr2']=0.01
def generate_command(opt):
    cmd = 'python train.py'
    for opt, val in opt.items():
        cmd += ' --' + opt + ' ' + str(val)
    return cmd

def run(opt):
    opt_ = copy.deepcopy(opt)
    os.system(generate_command(opt_))


# laList = [0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-0,
#           10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000]
# laList = [0, 0.01, 0.1, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 500, 1000, 10000, 50000, 100000, 500000, 700000, 900000]
#
# laList1 = [0, 0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9, 1, 3, 5, 7, 9, 10, 20, 30, 40, 50]
#
#
# for lamb1 in laList1:
#     for lamb in laList:
#         opt['lamb'] = lamb
#         opt['lamb1'] = lamb1
#         run(opt)
#
# laList = [0.1, 1, 10, 50, 100,500,1000,10000,50000]
# #
# laList1 = [0.01, 0.1, 1, 50, 100, 500]


# laList = [0.0001,0.001,0.01,0.1,1,10,50]
# lr1List = [0.0001,0.001,0.005,0.01,0.05,0.1]
# gcnList = [16,20,30,40,60,80]
# weigtdecay = [1e-4,5e-4,1e-3,5e-3,1e-2]


# flags.DEFINE_float('learning_rate', 0.005, 'Initial learning rate.') # 0.01
# flags.DEFINE_integer('epochs', 2000, 'Number of epochs to train.') # 200
# flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.') # 16
# flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).') # 0.5
# flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.') # 5e-4
# flags.DEFINE_integer('early_stopping', 100, 'Tolerance for early stopping (# of epochs).') # 10
# flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.') # 3
# flags.DEFINE_integer('pre', 1, 'Percentage of training set')
# flags.DEFINE_float('lamb', 1, 'lamb.')
#sname = ("featpix", "featzer", "featfac", "featfou", "featkar", "featmor")



# for nlr1List in lr1List:
#     for ngcnList in gcnList:
#         for nweigtdecay in weigtdecay:
#             for lamb in laList:
#                 for dname in sname:
#                     opt['learning_rate'] = nlr1List
#                     opt['hidden1'] = ngcnList
#                     opt['weight_decay'] = nweigtdecay
#                     opt['lamb'] = lamb
#                     run(opt)



# lr1List = [0.001,0.01,0.05,0.1]
# gcnList = [16,32,60,80,120,160]
# weigtdecay = [1e-4,5e-4]
# laList = [0.01,1, 50, 100, 200, 500,600,800,1000]
# flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.') # 0.05
# flags.DEFINE_integer('hidden1', 80, 'Number of units in hidden layer 1.') # 16
# flags.DEFINE_float('dropout', 0.6, 'Dropout rate (1 - keep probability).') # 0.5
# flags.DEFINE_float('weight_decay', 0.0001, 'Weight for L2 loss on embedding matrix.') # 5e-4
# flags.DEFINE_float('lamb', 50, 'lamb.')
sname = ("1", "2", "3", "4", "5", "6","7","8","9","10")
# lr1List = [0.001,0.01,0.05,0.1]
# gcnList = [16,32,60,80,120,160,180,200]
# weigtdecay = [1e-4,5e-4]
# laList = [0.01,1, 100, 500, 800,1000,1200,1500,2000]
#
# laList2 = [0.000000001,0.0000001,0.0001,0.01,1,10,100]
#
# laList3 = [0.000000001,0.0000001,0.0001,0.01,1,10,100]
#
# mlpList = [60,80,120,180,300,500]

mlpList = [30,40]

gcnList = [16,18,20]

# laList = [50,100,300,500,600]
#
# laList2 = [0.000000001,0.0000001,0.0001]
#
# laList3 = [0.000000001,0.0000001,0.0001]

# laList = [0.01,0.1,1, 10]
# laList2 = [0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001]
# laList3 = [0.000000001,0.00000001,0.0000001,0.000001,0.00001,0.0001]
# laList = [0.01,0.1,1]
# laList2 = [0.00000001,0.0000001,0.000001]
# laList3 = [0.000000001,0.00000001,0.0000001]

# for lamb3 in laList3:
#     for lamb2 in laList2:
#         for lamb in laList:
#             for nmlpList in mlpList:
#                 for ngcnList in gcnList:
#                     for dname in sname:
#                         opt['lamb'] = lamb
#                         opt['lamb2'] = lamb2
#                         opt['lamb3'] = lamb3
#                         opt['hidden_mlp'] = nmlpList
#                         opt['hidden1'] = ngcnList
#                         opt['dataset'] = dname
#                         run(opt)
# for lamb3 in laList3:
#     for lamb2 in laList2:
#         for lamb in laList:
#             for nmlpList in mlpList:
#                 for ngcnList in gcnList:
#                     for dname in sname:
#                         opt['lamb'] = lamb
#                         opt['lamb2'] = lamb2
#                         opt['lamb3'] = lamb3
#                         opt['hidden_mlp'] = nmlpList
#                         opt['hidden1'] = ngcnList
#                         opt['dataset'] = dname
#                         run(opt)
#
# laList = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10,100,1000,10000]
# laListS = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10,100]
# for lambS in laListS:
#     for lamb in laList:
#         opt['lamb'] = lamb
#         opt['lambS'] = lambS
#         run(opt)

# gcnList = [16,20,24,28,32,36,40]
# laList = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10,100,1000,10000]
# laListS = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10,100]
# laList3 = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10,100]
# for lamb3 in laList3:
#     for lambS in laListS:
#         for lamb in laList:
#             for ngcnList in gcnList:
#                 opt['lamb'] = lamb
#                 opt['lambS'] = lambS
#                 opt['lamb3'] = lamb3
#                 opt['hidden1'] = ngcnList
#                 run(opt)

gcnList = [24,30,32,34]
laList = [0.000000001,0.0000001,0.00001,0.001,0.1,1]
laListS = [0.000000001,0.0000001,0.00001,0.001,0.1,1,10]
laList3 = [0.000000001,0.0000001,0.00001,0.001,0.1,1]

for lamb3 in laList3:
    for lambS in laListS:
        for lamb in laList:
            for ngcnList in gcnList:
                for dname in sname:
                    opt['lamb'] = lamb
                    opt['lambS'] = lambS
                    opt['lamb3'] = lamb3
                    opt['hidden1'] = ngcnList
                    opt['dataset'] = dname
                    run(opt)