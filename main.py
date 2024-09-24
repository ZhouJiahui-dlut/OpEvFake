import torch
import torch.nn as nn
import numpy as np
import sys
import random
import torch.backends.cudnn as cudnn
import torchmetrics

from Diffusion.Multimodal_Diffusion import GaussianDiffusionTrainer
from train import train, valid
from dataloader_fakesv import get_dataloader
from eval_metrics import eval_FakeSV

from tensorboardX import SummaryWriter
writer = SummaryWriter("logs")


def main(model_config = None):
    modelConfig = {
        "state": "train",
        "epoch": 60,
        "batch_size": 16,
        "T": 100,
        "mult_dropout": 0.4,
        "Text_Pre_dropout": 0.2,
        "Img_Pre_dropout": 0.2,
        "lr": 5e-6,
        "beta_1": 1e-4,
        "beta_T": 0.02,
        "device": "cuda:0", ### MAKE SURE YOU HAVE A GPU !!!
        "t_in": 768,
        "i_in": 2048,
        "a_in": 128, #12288
        "v_in": 4096,
        "c3d_in": 4096,
        "t_in_pre": 100,
        "a_in_pre": 128,
        "v_in_pre": 1000,
        "c3d_in_pre": 128,
        "label_dim": 2,
        "d_m": 128,
        "unified_size": 128,
        "vertex_num": 32,
        "routing": 2,
        "T_t": 512,
        "T_i": 49,
        "T_a": 50,
        "T_v": 83,
        "early_stop": 5,
        "weight_decay": 0.99,
        "datamode": 'title+ocr', # title/title+ocr
        "comments_dropout": 0.3
    }
    if model_config is not None:
        modelConfig = model_config
    device = torch.device(modelConfig["device"])

    # data
    print("Start loading the data....")
    dataloader = get_dataloader(modelConfig=modelConfig, data_type='SVFEND')
    print('Finish loading the data....')

    # model
    trainer = GaussianDiffusionTrainer(
        modelConfig, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"],
        modelConfig["t_in"], modelConfig["a_in"], modelConfig["v_in"], modelConfig["d_m"], modelConfig["mult_dropout"],
        modelConfig["label_dim"],
        modelConfig["unified_size"], modelConfig["vertex_num"], modelConfig["routing"], modelConfig["T_t"],
        modelConfig["T_a"],  modelConfig["T_v"], modelConfig["batch_size"]).to(device)

    optimizer = torch.optim.AdamW(trainer.parameters(), lr=modelConfig["lr"], weight_decay=modelConfig["weight_decay"])
    criterion = nn.CrossEntropyLoss().to(device)

    if modelConfig["dataset"] in ['WEIBO']:
        best_valid_acc = -1
        epoch, best_epoch = 0, 0

    while True:
        epoch += 1
        # train for one epoch
        train_loss, train_results, train_truths = train(modelConfig, dataloader["train"], trainer, criterion, optimizer)
        train_acc, train_f1, train_pre, train_rec = eval_FakeSV(train_results, train_truths)
        # validate for one epoch
        valid_loss, valid_results, valid_truths = valid(dataloader["val"], trainer, criterion, modelConfig)
        valid_acc, valid_f1, valid_pre, valid_rec = eval_FakeSV(valid_results, valid_truths)
        # test for one epoch
        test_loss, test_results, test_truths = valid(dataloader["test"], trainer, criterion, modelConfig)
        test_acc, test_f1, test_pre, test_rec = eval_FakeSV(test_results, test_truths)

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        writer.add_scalar('train_f1', train_f1, epoch)
        writer.add_scalar('train_pre', train_pre, epoch)
        writer.add_scalar('train_rec', train_rec, epoch)
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        writer.add_scalar('test_f1', test_f1, epoch)
        writer.add_scalar('test_pre', test_pre, epoch)
        writer.add_scalar('test_rec', test_rec, epoch)
        writer.close()

        if modelConfig["dataset"] in ['mosi', 'mosei_senti', 'WEIBO']:
            print('Epoch {:2d} Loss| Train Loss{:5.4f} | Val Loss {:5.4f} | Test Loss{:5.4f} || Acc| Train Acc {:5.4f} | Val Acc {:5.4f} | Test Acc {:5.4f}'
                  .format(epoch, train_loss, valid_loss, test_loss, train_acc, valid_acc, test_acc))
            if valid_acc >= (best_valid_acc + 1e-4):
                print('{} dataset | acc improved! saving model to {}_best_model.pkl'
                      .format(modelConfig["dataset"], modelConfig["dataset"]))
                torch.save(trainer, '{}_best_model.pkl'.format(modelConfig["dataset"]))

                best_valid_acc = valid_acc
                best_acc = test_acc
                best_f1 = test_f1
                best_pre = test_pre
                best_rec = test_rec
                best_epoch = epoch
            if epoch - best_epoch >= modelConfig["early_stop"]:
                break
        if epoch > modelConfig["epoch"]:
            break

    if modelConfig["dataset"] in ['mosi', 'mosei_senti', 'WEIBO']:
        print("hyperparameter: ", modelConfig.items())
        print("Best Epoch:", best_epoch)
        print("Best Acc: {:5.4f}".format(best_acc))
        print("f1: {:5.4f}".format(best_f1))
        print("pre: {:5.4f}".format(best_pre))
        print("rec: {:5.4f}".format(best_rec))
        print('-' * 50)

# set seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

# log
class Logger(object):
    def __init__(self, filename='default.txt', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

if __name__ == '__main__':
    setup_seed(2021)
    sys.stdout = Logger('result.txt', sys.stdout)
    sys.stderr = Logger('error.txt', sys.stderr)
    main()
