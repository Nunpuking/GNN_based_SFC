# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function, division
from io import open
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn

def CountDown(sec):
    left_time = sec
    for i in range(sec):
        print("{}".format(left_time))
        time.sleep(1)
        left_time -= 1

def time_format(s):
    h = math.floor(s / 3600)
    m = math.floor((s-3600*h) / 60)
    s = s - h*3600 - m*60
    return '%dh %dm %ds' % (h, m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (time_format(s))

def ids2words(dict_map_inv, raw_data, sep=' ', eos_id=0, unk_sym='<unk>'):
    str_text = ''
    for vv in raw_data:
        if vv == eos_id:
            break
        if vv in dict_map_inv:
            str_text = str_text + sep + dict_map_inv[vv]
        else:
            str_text = str_text + sep + unk_sym
    return str_text.strip()

def unbpe(sentence):
    sentence = sentence.replace('<s>', '').strip()
    sentence = sentence.replace('</s>', '').strip()
    sentence = sentence.replace('@@ ', '')
    sentence = sentence.replace('@@', '')
    return sentence

def equizip(*iterables):
    iterators = [iter(x) for x in iterables]
    while True:
        try:
            first_value = iterators[0].__next__()
            try:
                other_values = [x.__next__() for x in iterators[1:]]
            except StopIteration:
                raise IterableLengthMismatch
            else:
                values = [first_value] + other_values
                yield tuple(values)
        except StopIteration:
            for iterator in iterators[1:]:
                try:
                    extra_value = iterator.__next__()
                except StopIteration:
                    pass # this is what we expect
                else:
                    raise IterableLengthMismatch
            raise StopIteration

def path_settings(subdir, running_mode, n_running):
    model_path = subdir + '/model' + str(n_running) + '.pth'
    if running_mode == 'train' and os.path.exists(model_path):
        print("WARNING : model_path {} already exists".format(model_path))
        CountDown(5)
    #    return 0
    #        n_running += 1
    #        model_path = subdir + '/model' + str(n_running) + '.pth'
    #    else:
    #        break
    if running_mode == 'train' or running_mode == 'pretrain':
        train_log_path = subdir + '/model' + str(n_running) + '.train.txt'
        valid_log_path = subdir + '/model' + str(n_running) + '.valid.txt'
        return model_path, train_log_path, valid_log_path
    else:
        return model_path

class training_manager():
    def __init__(self, args, decay_epochs=None):
        self.args = args
        self.n_patience = 0
        self.max_patience = args.patience
        self.n_lr_decay = 0
        self.max_lr_decay = args.lr_decay
        self.sch_decay = args.scheduled_lr_decay # bool
        self.sch_decay_epochs = decay_epochs
        self.current_lr = args.lr
        self.decay_ratio = args.lr_decay_ratio

    def patience_step(self, opt):
        self.n_patience += 1
        if self.n_patience >= self.max_patience and self.n_lr_decay >= self.max_lr_decay:
            return True, opt
        if self.sch_decay == 1:
            if self.n_lr_decay < self.max_lr_decay:
                self.n_patience = 0
        else:
            if self.n_patience >= self.max_patience:
                self.n_patience = 0
                opt = self.lr_decay_step(opt)
        return False, opt

    def epoch_step(self, epoch, opt):
        if self.sch_decay == 1 and epoch in self.sch_decay_epochs:
            self.sch_decay_epochs.remove(epoch)
            opt = self.lr_decay_step(opt)
        return opt

    def lr_decay_step(self, opt):
        self.n_lr_decay += 1
        self.current_lr *= self.decay_ratio

        for param_group in opt.param_groups:
            param_group['lr'] = self.current_lr
        return opt

def open_log(path, dir=False, message=False):
    if dir is True:
        if not os.path.exists(path):
            print("{} directory is made for {}".format(path, message))
            os.makedirs(path)
    else:
        if os.path.exists(path):
            print("Exist {} file is deleted".format(path))
            os.remove(path)
        print("{} file is opened for {}".format(path, message))
        log = open(path, 'a')
        return log

def weights_initializer(m):
    if isinstance(m, nn.Conv3d):
        nn.init.xavier_uniform_(m.weight.data, init.calculate_gain('relu'))
        # torch.nn.init.xavier_uniform_(m.bias.data)
    elif isinstance(m, nn.BatchNorm3d):
        m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data.fill_(0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        #nn.init.xavier_uniform(m.bias.data)
