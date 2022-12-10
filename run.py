import time
import torch
import torch.nn as nn
import argparse
import os
from torch import optim
from tensorboardX import SummaryWriter

from model.wdmcnet import WDMCNet
from utils.loaddata import load_train_data, load_test_data
from utils.split import split_data, get_test_set
from utils.cfg import train_transform, test_transform
from utils.toolbox import LOGGER, str2bool
from utils.initweights import init_weights, load_weights
from train import train
from eval import test, get_result_file
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime()) 


def main(config):
    # check tensorboard dir
    if not os.path.exists('./runs'):
        os.mkdir('./runs')
    if not os.path.exists('./weights'):
        os.mkdir('./weights')
    network_writer = SummaryWriter('runs/' + exp_time + '/network')
    train_loss_writer = SummaryWriter('runs/' + exp_time + '/train_loss')
    train_acc_writer = SummaryWriter('runs/' + exp_time + '/train_acc')
    train_dts_writer = SummaryWriter('runs/' + exp_time + '/train_dts')
    writer_group = {"loss": train_loss_writer, "acc": train_acc_writer, "dts": train_dts_writer}

    # load data
    if config.stage == 'self':
        get_test_set(config.rawpath, config.newpath)
        split_data(config.newpath, config.datadir)
    else:
        split_data(config.rawpath, config.datadir)
    train_dataset_path = os.path.join(config.datadir, config.dataset)
    test_dataset_path = config.newpath
    train_dataloader, valid_dataloader = load_train_data(config.trainscl, train_dataset_path, train_transform, config.bts)
    test_dataloader = load_test_data(test_dataset_path, test_transform)
    
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = 'cpu'
    print(f"Using {device} device")
    model = WDMCNet()
    print(model)
    
    dummy_input = torch.randn(1,1,52,52)
    network_writer.add_graph(model, (dummy_input, ), True)
    network_writer.close()
    
    if config.target == 'train':
        if config.initmodel:
            model = init_weights(model)
        if config.loadwt:
            model = load_weights(model, config.weights)
    elif config.target == 'eval' or config.target == 'eval_save':
        model = load_weights(model, config.weights)
    # model = nn.DataParallel(model)
    model = model.to(device)
    
    # set optim and loss fn
    if config.initmodel:
        if config.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0005)
    else:
        if config.optim == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=0.0005)
        elif config.optim == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005)
    loss_fn = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=False, threshold=1e-4, threshold_mode='rel', cooldown=3, min_lr=1e-7)
    if config.target == 'train':
        # train
        for epoch_idx in range(config.epoch):
            print(f"Epoch {epoch_idx+1}\n-------------------------------")
            train(model, train_dataloader, valid_dataloader, optimizer, loss_fn, device, writer_group, epoch_idx, scheduler)
        
        # test
        test(test_dataloader, model, device, loss_fn)
        torch.save(model.state_dict(), config.saveweights)
        print("Done!")
    elif config.target == 'eval':
        test(test_dataloader, model, device, loss_fn)
    elif config.target == 'eval-save':
        get_result_file(test_dataloader, model, device, config.result)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default='train', help="train or eval")
    parser.add_argument("--stage", type=str, default='self', help="project stage")
    parser.add_argument("--rawpath", type=str, default='./data/raw/datasets2022.npz', help="raw data path")
    parser.add_argument("--newpath", type=str, default='./data/raw/dataset_new.npz', help="new data path")
    parser.add_argument("--datadir", type=str, default='./data/processed', help="processed data path")
    parser.add_argument("--trainscl", type=float, default=0.7, help='train dataset scale')
    parser.add_argument("--bts", type=int, default=64, help='batch size')
    parser.add_argument("--lr", type=float, default=1e-3, help='learning rate')
    parser.add_argument("--initmodel", type=str2bool, default=False, help='init the model weights')
    parser.add_argument("--loadwt", type=str2bool, default=False, help='load model weights')
    parser.add_argument("--weights", type=str, default=None, help="load model weights path")
    parser.add_argument("--saveweights", type=str, default=None, help="save model weights path")
    parser.add_argument("--dataset", type=str, default='base.npz', help="using dataset")
    parser.add_argument("--epoch", type=int, default=5, help="epoch num")
    parser.add_argument("--optim", type=str, default='adam', help="optimizer(adam or sgd)")
    parser.add_argument("--result", type=str, default='./data/result/result.npz', help="result file path")
    args = parser.parse_args()
    main(args)
    