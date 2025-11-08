import torch 
import torch.nn as nn

def adjust_learning_rate(optimizer, epoch, lr_schedule):
    for schedule in lr_schedule:
        start, end = schedule["epoch_range"]
        if start <= epoch <= end:
            lr = schedule["learning_rate"]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            break
