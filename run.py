"""
@ File Name     :   run.py
@ Time          :   2022/12/13
@ Author        :   Cheng Kaiyue
@ Version       :   1.0
@ Contact       :   chengky18@icloud.com
@ Description   :   main function
@ Function List :   main() -- main function
"""


import argparse
import os
import time

import torch
import torch.nn as nn
from eval import get_result_file, get_result_file_raw, test
from linformer import Linformer
from model.wdmcnet import WDmCNetNeck, WDmCNetResNet, WDmCNetTransformer, WDmCNetVGG
from tensorboardX import SummaryWriter
from torch import optim
from train import train
from utils.cfg import *
from utils.initweights import init_weights, load_weights
from utils.loaddata import load_raw_test_data, load_test_data, load_train_data
from utils.split import get_test_set, split_data
from utils.toolbox import LOGGER, str2bool

# GPU ID
# default: 0
# if you have multiple GPUs and want to select the specific one, set this num
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# record the time when the program starting
exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())


def main(config):
    """main entry

    Args:
        config (args): args list
    """

    LOGGER.warning("Start")

    # xvgg16 xresnet50 xvit
    #
    # because the input size of vgg16 is 1x52x52
    # and the input size of resnet50 and vit is 1x224x224
    #
    # so we need different transform, and load different model
    LOGGER.info("Create Model")
    if config.model == "xvgg16":
        # vgg16
        train_transform = train_transform_vgg
        test_transform = test_transform_vgg
        model = WDmCNetVGG()
    elif config.model == "xresnet50":
        # resnet50
        train_transform = train_transform_resnet
        test_transform = test_transform_resnet
        model = WDmCNetResNet()
    elif config.model == "xvit":
        # transformer (vit)
        train_transform = train_transform_vit
        test_transform = test_transform_vit
        efficient_transformer = Linformer(
            dim=128, seq_len=49 + 1, depth=12, heads=8, k=64
        )
        model = WDmCNetTransformer(
            transformer=efficient_transformer,
        )
    elif config.model == "neck":
        # final neck
        #
        # load 3 base models and remove the last layer

        LOGGER.info("Init Neck-Model")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # load neck model
        train_transforms = {
            "raw": train_transform_vgg,
            "resize": train_transform_resnet,
        }
        test_transforms = {"raw": test_transform_vgg, "resize": test_transform_resnet}

        # load vgg16
        model_vgg = WDmCNetVGG()
        load_weights(model_vgg, "./weights/xvgg16/model-full-sgd.pth")
        model_vgg.classifier[-1] = nn.Sequential()
        model_vgg.classifier[-2] = nn.Sequential(nn.LeakyReLU(inplace=True))
        for _, param in enumerate(model_vgg.parameters()):
            param.requires_grad = False

        # load resnet50
        model_resnet = WDmCNetResNet()
        load_weights(model_resnet, "./weights/xresnet50/model-full-sgd.pth")
        model_resnet.classifier[-1] = nn.Sequential()
        model_resnet.classifier[-2] = nn.Sequential(nn.LeakyReLU(inplace=True))
        for _, param in enumerate(model_resnet.parameters()):
            param.requires_grad = False

        # load vit
        efficient_transformer = Linformer(
            dim=128, seq_len=49 + 1, depth=12, heads=8, k=64
        )
        model_vit = WDmCNetTransformer(
            transformer=efficient_transformer,
        )
        load_weights(model_vit, "./weights/xvit/model-full-sgd.pth")
        model_vit.mlp_head[-1] = nn.Sequential()
        model_vit.mlp_head[-2] = nn.Sequential(nn.LeakyReLU(inplace=True))
        for _, param in enumerate(model_vit.parameters()):
            param.requires_grad = False

        base_models = {
            "xvgg16": model_vgg,
            "xresnet50": model_resnet,
            "xvit": model_vit,
        }
        model = WDmCNetNeck(base_models)

    LOGGER.info("Init Path")
    # check tensorboard dir
    if not os.path.exists("./runs"):
        os.mkdir("./runs")
    if not os.path.exists(os.path.join("./runs", config.model)):
        os.mkdir(os.path.join("./runs", config.model))
    if not os.path.exists(config.weightsroot):
        os.mkdir(config.weightsroot)
    if not os.path.exists(os.path.join(config.weightsroot, config.model)):
        os.mkdir(os.path.join(config.weightsroot, config.model))
    print("init weights path -> {}".format(os.path.join("./runs", config.model)))

    # tensorboard writer
    train_writer = SummaryWriter(
        os.path.join("./runs", config.model, exp_time, "train")
    )
    valid_writer = SummaryWriter(
        os.path.join("./runs", config.model, exp_time, "valid")
    )
    writer_group = {"train": train_writer, "valid": valid_writer}
    print(
        "Init tensorboard path -> {}".format(
            os.path.join("./runs", config.model, exp_time)
        )
    )

    LOGGER.info("Process DataSet")
    # make fake dataset (train + test)
    # split dataset according to the num of defects
    if config.stage == "self":
        get_test_set(config.rawpath, config.newpath)
        split_data(config.newpath, config.datadir)
    elif config.stage == "raw-train":
        split_data(config.rawpath, config.datadir)

    # load dataset
    if config.stage == "final-test":
        # final test
        # dataset file -> list [data1, data2, data3, ...]
        if config.model != "neck":
            final_dataloader = load_raw_test_data(config.rawpath, test_transform, False)
        elif config.model == "neck":
            final_dataloader = load_raw_test_data(config.rawpath, test_transforms, True)
    else:
        # others: train and self test (using fake dataset)
        # dataset file -> dataloader provided by PyTorch
        train_dataset_path = os.path.join(config.datadir, config.dataset)
        test_dataset_path = config.newpath

        # the differences between neck and others is:
        # 1. transform
        # 2. dataloader
        if config.model != "neck":
            train_dataloader, valid_dataloader = load_train_data(
                config.trainscl,
                train_dataset_path,
                train_transform,
                config.bts,
                is_neck=False,
            )
            test_dataloader = load_test_data(
                test_dataset_path, test_transform, is_neck=False
            )
        elif config.model == "neck":
            train_dataloader, valid_dataloader = load_train_data(
                config.trainscl,
                train_dataset_path,
                train_transforms,
                config.bts,
                is_neck=True,
            )
            test_dataloader = load_test_data(
                test_dataset_path, test_transforms, is_neck=True
            )

    LOGGER.info("Set Device")
    # load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    LOGGER.info("Model Framework")
    print(model)

    # draw the model using tensorboard
    if config.initmodel:
        LOGGER.info("Draw the Model by Tensorboard")
        print(
            "network -> {}".format(
                os.path.join("./runs", config.model, exp_time, "network")
            )
        )
        network_writer = SummaryWriter(
            os.path.join("./runs", config.model, exp_time, "network")
        )
        if config.model == "neck":
            dummy_input = {
                "raw": torch.randn(1, 1, 52, 52),
                "resize": torch.randn(1, 1, 224, 224),
            }
        elif config.model == "xvgg16":
            dummy_input = torch.randn(1, 1, 52, 52)
        else:
            dummy_input = torch.randn(1, 1, 224, 224)
        network_writer.add_graph(model, (dummy_input,), True)
        network_writer.close()

    # rebuild the neck model
    # @remark: tensorboard only accept data in CPU device
    #           so we can't use .to(decive) before draw the model by tensorboard
    #           after drawing, we rebuild the model and put it in device
    if config.model == "neck":
        LOGGER.info("Rebuild the Neck-Model")
        model_vgg.eval()
        model_resnet.eval()
        model_vit.eval()
        base_models = {
            "xvgg16": model_vgg.to(device),
            "xresnet50": model_resnet.to(device),
            "xvit": model_vit.to(device),
        }
        model = WDmCNetNeck(base_models)

    # load weights or init weights
    LOGGER.info("Process Weight")
    if config.target == "train":
        if config.initmodel:
            model = init_weights(model)
        if config.loadwt:
            model = load_weights(
                model, os.path.join(config.weightsroot, config.model, config.weights)
            )
    elif config.target == "eval" or config.target == "eval_save":
        model = load_weights(
            model, os.path.join(config.weightsroot, config.model, config.weights)
        )

    model = model.to(device)

    # set optimizer, loss function and learning rate scheduler
    LOGGER.info("Set optimizer, loss function and learning rate scheduler")
    if config.initmodel:
        if config.optim == "adam":
            print("Using Adam optim")
            optimizer = optim.Adam(
                model.parameters(), lr=config.lr, weight_decay=0.0005
            )
            if config.model == "xvit":
                optimizer = optim.Adam(model.parameters(), lr=config.lr)
    else:
        if config.optim == "adam":
            print("Using Adam optim")
            optimizer = optim.Adam(
                model.parameters(), lr=config.lr, weight_decay=0.0005
            )
            if config.model == "xvit":
                optimizer = optim.Adam(model.parameters(), lr=config.lr)
        elif config.optim == "sgd":
            print("Using SGD optim")
            optimizer = optim.SGD(
                model.parameters(), lr=config.lr, momentum=0.9, weight_decay=0.0005
            )
            if config.model == "xvit":
                optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9)
            if config.final:
                optimizer = optim.SGD(
                    model.parameters(), lr=config.lr, weight_decay=0.000005
                )
    loss_fn = nn.MSELoss()
    print("Using MSE Loss")
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        "min",
        factor=0.1,
        patience=4,
        verbose=False,
        threshold=1e-4,
        threshold_mode="rel",
        cooldown=3,
        min_lr=1e-7,
    )
    print("Using ReduceLROnPlateau lr_scheduler")

    # test, train or eval
    if config.stage == "final-test":
        # test, get final submit file (.csv)
        LOGGER.info("Final Test (get result file)")
        get_result_file_raw(
            final_dataloader, model, device, config.result, (config.model == "neck")
        )
        pass
    else:
        if config.target == "train":
            # train
            LOGGER.info("Train")
            for epoch_idx in range(config.epoch):
                print(f"Epoch {epoch_idx+1}\n-------------------------------")
                train(
                    model,
                    train_dataloader,
                    valid_dataloader,
                    optimizer,
                    loss_fn,
                    device,
                    writer_group,
                    epoch_idx,
                    scheduler,
                    is_neck=(config.model == "neck"),
                )

            # test
            LOGGER.info("Test")
            test(
                test_dataloader,
                model,
                device,
                loss_fn,
                is_neck=(config.model == "neck"),
            )
            torch.save(
                model.state_dict(),
                os.path.join(config.weightsroot, config.model, config.saveweights),
            )
            LOGGER.info(
                "Save model -> {}".format(
                    os.path.join(config.weightsroot, config.model, config.saveweights)
                )
            )
            print("Done!")
        elif config.target == "eval":
            # eval
            LOGGER.info("Eval")
            test(
                test_dataloader,
                model,
                device,
                loss_fn,
                is_neck=(config.model == "neck"),
            )
        elif config.target == "eval-save":
            # eval (deprecated)
            get_result_file(
                test_dataloader,
                model,
                device,
                config.result,
                is_neck=(config.model == "neck"),
            )
    LOGGER.warning("Done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default="train", help="train or eval")
    parser.add_argument(
        "--stage",
        type=str,
        default="self",
        help="project stage, self/raw-train/final-test",
    )
    parser.add_argument(
        "--rawpath",
        type=str,
        default="./data/raw/datasets2022.npz",
        help="raw dataset path",
    )
    parser.add_argument(
        "--newpath",
        type=str,
        default="./data/raw/dataset_new.npz",
        help="new dataset path",
    )
    parser.add_argument(
        "--datadir", type=str, default="./data/processed", help="processed dataset path"
    )
    parser.add_argument(
        "--trainscl", type=float, default=0.7, help="train dataset scale"
    )
    parser.add_argument("--bts", type=int, default=64, help="batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument(
        "--initmodel",
        type=str2bool,
        default=False,
        help="init the model weights (True/False)",
    )
    parser.add_argument(
        "--loadwt", type=str2bool, default=False, help="load model weights (True/False)"
    )
    parser.add_argument(
        "--weightsroot",
        type=str,
        default="./weights/",
        help="load model weights root path",
    )
    parser.add_argument(
        "--weights", type=str, default=None, help="load model weights path"
    )
    parser.add_argument(
        "--model", type=str, default="xvgg16", help="model type (xvgg16/xresnet50/xvit)"
    )
    parser.add_argument(
        "--saveweights", type=str, default=None, help="save model weights path"
    )
    parser.add_argument("--dataset", type=str, default="full.npz", help="using dataset")
    parser.add_argument("--epoch", type=int, default=5, help="epoch num")
    parser.add_argument(
        "--optim", type=str, default="adam", help="optimizer (adam or sgd)"
    )
    parser.add_argument(
        "--result",
        type=str,
        default="./data/result/result.npz",
        help="result file path",
    )
    parser.add_argument(
        "--final", type=str2bool, default=False, help="is the last one (True/False)"
    )
    args = parser.parse_args()
    main(args)
