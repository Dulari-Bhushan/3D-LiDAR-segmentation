'''
The MIT License
Copyright (c) 2019 Tiago Cortinhal (Halmstad University, Sweden), George Tzelepis (Volvo Technology AB, Volvo Group Trucks Technology, Sweden) and Eren Erdal Aksoy (Halmstad University and Volvo Technology AB, Sweden)
Copyright (c) 2019 Andres Milioto, Jens Behley, Cyrill Stachniss, Photogrammetry and Robotics Lab, University of Bonn.

References:
https://github.com/PRBonn/lidar-bonnetal
https://github.com/TiagoCortinhal/SalsaNext
'''

import numpy as np
import torch


class IOUEval:
    def __init__(self, n_classes, device=torch.device('cpu'), ignore=None, is_distributed=False):
        self.n_classes = n_classes
        self.device = device
        # if ignore is larger than n_classes, consider no ignoreIndex
        self.ignore = torch.tensor(ignore).long()
        self.include = torch.tensor(
            [n for n in range(self.n_classes) if n not in self.ignore]).long()
        print('[IOU EVAL] IGNORE: ', self.ignore)
        print('[IOU EVAL] INCLUDE: ', self.include)
        self.is_distributed = is_distributed
        self.reset()

    def num_classes(self):
        return self.n_classes

    def reset(self):
        self.conf_matrix = torch.zeros(
            (self.n_classes, self.n_classes), device=self.device).long()
        self.ones = None
        self.last_scan_size = None  # for when variable scan size is used
        self._reduced_conf = None  # cache for the cross-rank reduced matrix

    def addBatch(self, x, y):  # x=preds, y=targets
        # if numpy, pass to pytorch to tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.array(x)).long().to(self.device)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(np.array(y)).long().to(self.device)

        # sizes should be 'batch_size x H x W'
        x_row = x.reshape(-1)  # de-batchify
        y_row = y.reshape(-1)  # de-batchify

        # idxs are labels and predictions
        idxs = torch.stack([x_row, y_row], dim=0)

        # ones is what I want to add to conf when I
        if self.ones is None or self.last_scan_size != idxs.shape[-1]:
            self.ones = torch.ones((idxs.shape[-1]), device=self.device).long()
            self.last_scan_size = idxs.shape[-1]

        # make confusion matrix (cols = gt, rows = pred)
        self.conf_matrix = self.conf_matrix.index_put_(
            tuple(idxs), self.ones, accumulate=True)
        self._reduced_conf = None  # matrix changed, invalidate the cached reduction

    def synchronize(self):
        '''
        Reduce the confusion matrix across ranks exactly once and cache the result.

        Previously every getter called getStats(), which performed
        barrier + all_reduce + barrier on its own. That meant one collective per
        log step during training plus 3-4 more at each epoch end. Combined with the
        default 30 min NCCL watchdog it is the direct cause of the timeouts seen on
        full-length sequence-08 validation. Call this once per epoch, then read the
        getters freely.
        '''
        if not self.is_distributed:
            self._reduced_conf = self.conf_matrix.clone().double()
            return
        conf_gpu = self.conf_matrix.clone().double().cuda()
        # all_reduce is itself a synchronizing collective; the surrounding barriers
        # were redundant.
        torch.distributed.all_reduce(conf_gpu)
        self._reduced_conf = conf_gpu.to(self.conf_matrix.device)
        del conf_gpu

    def getStats(self, local=False):
        '''
        local=True reads this rank's own matrix and never touches NCCL. Use it for
        intra-epoch progress logging; use the synchronized path for reported numbers.
        '''
        if local or not self.is_distributed:
            conf = self.conf_matrix.clone().double()
        else:
            if self._reduced_conf is None:
                self.synchronize()
            conf = self._reduced_conf.clone()

        # remove fp and fn from confusion on the ignore classes cols and rows
        conf[self.ignore] = 0
        conf[:, self.ignore] = 0

        # get the clean stats
        tp = conf.diag()
        fp = conf.sum(dim=1) - tp
        fn = conf.sum(dim=0) - tp
        return tp, fp, fn

    def getIoU(self, local=False):
        tp, fp, fn = self.getStats(local=local)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()
        return iou_mean, iou  # returns 'iou mean', 'iou per class' ALL CLASSES

    def getIoUnAcc(self, local=False):
        tp, fp, fn = self.getStats(local=local)
        intersection = tp
        union = tp + fp + fn + 1e-15
        iou = intersection / union
        iou_mean = (intersection[self.include] / union[self.include]).mean()

        total = tp + fp + 1e-15
        acc = tp / total
        acc_mean = acc[self.include].mean()

        return iou_mean, iou, acc_mean, acc  # returns 'iou mean', 'iou per class' ALL CLASSES

    def getAcc(self, local=False):
        tp, fp, fn = self.getStats(local=local)
        total = tp + fp + 1e-15
        acc = tp / total
        acc_mean = acc[self.include].mean()
        return acc_mean, acc

    def getRecall(self, local=False):
        tp, fp, fn = self.getStats(local=local)
        total = tp + fn + 1e-15
        recall = tp / total
        recall_mean = recall[self.include].mean()
        return recall_mean, recall
