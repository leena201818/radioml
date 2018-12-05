import sys
import argparse
import os
from keras.models import model_from_json

from rml2018_01 import predict,plot_tSNE

def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model', type=str,
                        help='无线电信号调制识别的模型：vgg,resnet,cldnn.')
    parser.add_argument('testfile', type=str,
                        help='测试数据文件路径.')

    parser.add_argument('-acc', '--accuracy',
                        help='显示正确率曲线', action='store_true')

    group = parser.add_argument_group('comfusion matrix')
    group.add_argument('-conf','--confusion_matrix',
                        help='显示混淆矩阵',action='store_true')
    group.add_argument('-snr','--min_snr',
                        help='混淆矩阵显示的最小信噪比.',type=int,default=-10)

    parser.add_argument('-tsne','--intermediate_representation',
                        help='采用t-SNE绘制网络中间表示层.',action='store_true')
    return parser.parse_args(argv)

if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    if not os.path.exists(args.testfile) :
        print('输入的测试文件路径不正确！')
        exit(0)
    model_config_file = 'models/vgg-like-1024-1m-tap8.wts-58.9p.json'
    weight_file = 'weights/good/vgg-like-1024-1m-tap8.wts-58.9p.h5'
    if args.model == 'vgg':
        model_config_file = 'models/vgg-like-1024-1m-tap8.wts-58.9p.json'
        weight_file = 'weights/good/vgg-like-1024-1m-tap8.wts-58.9p.h5'
    if args.model == 'resnet':
        model_config_file = 'models/res-like-1024-1m-tap8_L6.wts-58.4.json'
        weight_file = 'weights/good/res-like-1024-1m-tap8_L6.wts-58.4.h5'
    if args.model == 'cldnn':
        model_config_file = 'models/cldnn-like-1024-512k-L4-2LSTM-55.9p.wts.json'
        weight_file = 'weights/good/cldnn-like-1024-512k-L4-2LSTM-55.9p.wts.h5'


    with open(model_config_file,'r') as f:
        model = model_from_json(f.read())
        model.summary()

        print(args.min_snr)

        predict(model,weight_file=weight_file,test_filename=args.testfile,dis_acc=args.accuracy,dis_conf=args.confusion_matrix,min_snr=args.min_snr)
        if args.intermediate_representation:
            testfile = '/media/XYZ_1024_128k.hdf5'
            print(testfile)
            plot_tSNE(model,weight_file=weight_file,test_filename=testfile)

