import torch.backends.cudnn as cudnn
from version_pytorch.u_net_pytorch.net.unet import Unet
from version_pytorch.u_net_pytorch.train_model.dataloader import *
import time
import os
from torch import optim, nn
import torch
from torch.utils.data import DataLoader
import argparse


def calDice(y_pred, y_true):
    smooth = 1.
    y_true_f = y_true.ravel()
    y_pred_f = y_pred.ravel()
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f * y_true_f) + np.sum(y_pred_f * y_pred_f) + smooth)


def calAccuracy(predict, label):
    """
    计算Accuracy，正确点数量
    """
    predict = np.array(predict)
    label = np.array(label)
    true_point = predict == label
    true_num = len(label[true_point])
    Accuracy = true_num / (label.shape[0] * label.shape[1] * label.shape[2])

    return Accuracy


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)

    if classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)


def train(train_loader, cfg):
    print("Newly Training!")
    net = Unet(1, 1).to(cfg.device)
    net.apply(weights_init_xavier)  # 权值初始化
    cudnn.benchmark = True

    criterion_mse = torch.nn.BCELoss().to(cfg.device)

    # 优化函数，优化器
    optimizer = optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)
    k = 0
    min_loss = 5.0
    for idx_epoch in range(cfg.n_epochs):
        iter_loss_accum = 0.0
        loss_list = []
        dice_list = []
        i = 0
        net.train()
        for index, data in enumerate(train_loader):
            k += 1
            time_start = time.time()
            """这里修改cpu，gpu版本"""
            train = data[0].to(cfg.device)
            label = data[1].to(cfg.device)
            predict_label = net(train)
            entropy_loss = criterion_mse(predict_label.reshape(label.shape[0], label.shape[2], label.shape[3]),
                                         label.reshape(label.shape[0], label.shape[2], label.shape[3]))

            pred_label = predict_label.cpu().detach().numpy()
            gt_label = label.cpu().detach().numpy()
            dice = calDice(pred_label, gt_label)
            loss = entropy_loss

            iter_loss_accum += loss.data
            loss_list.append(loss.cpu().detach().numpy())
            dice_list.append(dice)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            i += 1

            pred_label = np.reshape(pred_label, (pred_label.shape[0], 400, 400))
            pred_label *= 255
            pred_label = pred_label.astype('uint8')
            pred_label[pred_label > 50] = 255
            pred_label[pred_label <= 50] = 0

            gt_label = np.reshape(gt_label, (gt_label.shape[0], 400, 400))
            gt_label *= 255
            gt_label = gt_label.astype('uint8')

            accuracy = calAccuracy(pred_label, gt_label)

            print("\r", "Epoch: [%d/%d], Iteration: [%d/%d], loss: %.4f, men loss: %.4f, "
                        "dice: %.4f, mean dice: %.4f, accuracy: %.4f,time: %.4f" %
                  (idx_epoch, cfg.n_epochs, i, len(train_loader), float(loss.data.cpu()), float(np.mean(loss_list)),
                   float(dice), float(np.mean(dice_list)), float(accuracy), time.time() - time_start), end="")
        print(" epoch:{}, mean loss:{:.4f}, mean dice:{:4f}".format(idx_epoch, np.mean(loss_list), np.mean(dice_list)))
        if np.mean(loss_list) < min_loss:
            min_loss = np.mean(loss_list)
            if not os.path.isdir("../model_" + cfg.model):
                os.makedirs("../model_" + cfg.model)
            torch.save(net.state_dict(), "../model_" + cfg.model + "/model_" + str(idx_epoch + 40) + ".pkl")
            print(" model saved! \n")
        else:
            print(" model not saved! \n")
        scheduler.step()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str,
                        default=r"D:\code_project\Python_project\paper_project_1"
                                r"\UV_Net_paper\version_keras\u_net\dataset\data_cv_clip\train")
    parser.add_argument('--label_dir', type=str,
                        default=r"D:\code_project\Python_project\paper_project_1"
                                r"\UV_Net_paper\version_keras\u_net\dataset\data_cv_clip\label")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epochs', type=int, default=50)
    # 线程
    parser.add_argument('--num_works', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--gamma', type=float, default=0.9, help='')
    parser.add_argument('--n_steps', type=int, default=10, help='number of epochs to update learning rate')
    parser.add_argument('--load_checkpoint', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')

    return parser.parse_args()


def main(cfg):
    trainset = DataFeeder(train_dir=cfg.train_dir, label_dir=cfg.label_dir)
    train_loader = DataLoader(dataset=trainset, num_workers=cfg.num_works,
                              batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    train(train_loader, cfg)


if __name__ == '__main__':
    configures = parse_args()
    main(configures)
