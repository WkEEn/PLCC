import math
import shutil
import logging
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch
from sklearn import metrics
from sklearn.preprocessing import label_binarize


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class ConfusionMatrix(object):
    def __init__(self, classes):
        self.confusion_matrix = torch.zeros(len(classes), len(classes))
        self.classes = classes

    def update_matrix(self, preds, targets):
        preds = torch.max(preds, 1)[1].cpu().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.confusion_matrix[t, p] += 1

    def plot_confusion_matrix(self, normalize=True, save_path='./Confusion Matrix.jpg'):
        cm = self.confusion_matrix.numpy()
        classes = self.classes
        num_classes = len(classes)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')
        im = plt.matshow(cm, cmap=plt.cm.Blues)  # cm.icefire
        plt.xticks(range(num_classes), classes, fontproperties="Times New Roman")
        plt.yticks(range(num_classes), classes, fontproperties="Times New Roman")
        ax = plt.gca()
        ax.xaxis.set_ticks_position('bottom')
        for i in range(len(classes)):
            tempSum = 0
            for j in range(num_classes - 1):
                tempS = cm[i, j] * 100
                tempSum += tempS
                color = 'white' if tempS > 50 else 'black'
                if cm[i, j] != 0:
                    plt.text(j, i, format(tempS, '0.2f'), color=color, ha='center')
            tempS = 100 - tempSum
            color = 'white' if tempS > 50 else 'black'
            if float(format(abs(tempS), '0.2f')) != 0:
                plt.text(num_classes - 1, i, format(abs(tempS), '0.2f'), color=color, ha='center')

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="2%", pad=0.1)
        cb = plt.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5)
        cb.set_ticks(np.linspace(0, 1, 6))
        cb.set_ticklabels(('0', '20', '40', '60', '80', '100'))

        plt.savefig(save_path)
        plt.close()


class AUCMetric(object):
    def __init__(self, classes):
        self.targets = []
        self.preds = []
        self.classes = np.arange(len(classes))

    def update(self, preds, targets):
        preds = torch.softmax(preds.cpu(), dim=-1).detach().numpy()
        targets = targets.cpu().numpy()
        for p, t in zip(preds, targets):
            self.preds.append(p)
            self.targets.append(t)

    def calc_auc_score(self):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), classes=self.classes)
        micro_auc = metrics.roc_auc_score(targets, preds, average='micro')
        macro_auc = metrics.roc_auc_score(targets, preds, average='macro')
        return micro_auc, macro_auc

    def plot_roc_curve(self, save_path):
        preds = np.array(self.preds)
        targets = label_binarize(np.array(self.targets), self.classes)
        fpr, tpr, thresholds, = metrics.roc_curve(targets.ravel(), preds.ravel())
        auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr, c='r', lw=2, alpha=0.7, label='AUC={:.3f}'.format(auc))
        plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)
        plt.xlim((-0.01, 1.02))
        plt.ylim((-0.01, 1.02))
        plt.xticks(np.arange(0, 1.1, 0.1))
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.xlabel('False Positive Rate', fontsize=13)
        plt.ylabel('True Positive Rate', fontsize=13)
        plt.grid(b=True, ls=':')
        plt.legend(loc='lower right', fancybox=True, framealpha=0.8, fontsize=12)
        plt.savefig(save_path)
        plt.close()
