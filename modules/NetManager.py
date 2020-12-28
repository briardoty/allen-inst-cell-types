#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:19:48 2020

@author: briardoty
"""
import torch
import gc
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import math
import time
import seaborn as sns
import numpy as np
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import random
import matplotlib.pyplot as plt
import functools
import sys

try:
    from .MixedActivationLayer import MixedActivationLayer
except:
    from MixedActivationLayer import MixedActivationLayer

try:
    from .StickNet import StickNet
except:
    from StickNet import StickNet

try:
    from .util import *
except:
    from util import *

from enum import Enum
class TargetLayers(Enum):
    NOTHING = 0
    FIRSTHALF = 1
    LASTHALF = 2
    ALL = None

def replace_act_layers(model, n_repeat, act_fns, act_fn_params, spatial,
    conv_target, fc_target, default_fn):
    """
    Iterative helper function to replace relu layers with
    instances of MixedActivationLayer with the given params
    """

    # replace activation layers in features (conv) group
    conv_layers = list(model.features._modules.items())
    n_tgts = sum([1 if type(m) == nn.ReLU else 0 for n, m in conv_layers])
    conv_target = TargetLayers(conv_target)
    if conv_target in [TargetLayers.FIRSTHALF, TargetLayers.LASTHALF]:
        n_tgts = n_tgts / 2
    elif conv_target == TargetLayers.NOTHING:
        n_tgts = 0
    itr = (reversed(range(len(conv_layers))) 
        if conv_target == TargetLayers.LASTHALF 
        else range(len(conv_layers)))
    for i in itr:

        name, module = conv_layers[i]

        if type(module) != nn.ReLU:
            continue

        _, prev = conv_layers[i-1]
        n_features = prev.out_channels

        if n_tgts <= 0:
            # replace with default component layer
            model.features._modules[name] = MixedActivationLayer(n_features, [1], 
                [act_fns[default_fn]], [act_fn_params[default_fn]], spatial) 

        else:
            # replace with mixed layer
            model.features._modules[name] = MixedActivationLayer(n_features, n_repeat, 
                act_fns, act_fn_params, spatial) 
            
            n_tgts -= 1

    # then do the same thing for fc layers
    fc_layers = list(model.classifier._modules.items())
    n_tgts = sum([1 if type(m) == nn.ReLU else 0 for n, m in fc_layers])
    fc_target = TargetLayers(fc_target)
    if fc_target in [TargetLayers.FIRSTHALF, TargetLayers.LASTHALF]:
        n_tgts = n_tgts / 2
    elif fc_target == TargetLayers.NOTHING:
        n_tgts = 0
    itr = reversed(range(len(fc_layers))) if fc_target == TargetLayers.LASTHALF else range(len(fc_layers))
    for i in itr:

        name, module = fc_layers[i]

        if type(module) != nn.ReLU:
            continue

        _, prev = fc_layers[i-1]
        n_features = prev.out_features
        
        if n_tgts <= 0:
            # replace with default component layer
            model.classifier._modules[name] = MixedActivationLayer(n_features, [1], 
                [act_fns[default_fn]], [act_fn_params[default_fn]], spatial) 

        else:
            # replace with mixed layer
            model.classifier._modules[name] = MixedActivationLayer(n_features, n_repeat, 
                act_fns, act_fn_params, spatial) 
            
            n_tgts -= 1

    return model


class NetManager():
    
    def __init__(self, dataset, net_name, group, case, data_dir, 
        train_scheme, pretrained=False, seed=None):

        # SEED!
        self.seed_everything(seed)

        self.dataset = dataset
        self.net_name = net_name
        self.group = group
        self.case = case
        self.data_dir = os.path.expanduser(data_dir)
        self.train_scheme = train_scheme
        self.pretrained = pretrained
        self.epoch = 0
        self.modified_layers = None
        self.initial_lr = None
        self.perf_stats = []

        if (torch.cuda.is_available()):
            print("Enabling GPU speedup!")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("GPU speedup NOT enabled.")
        
    def seed_everything(self, seed):
        
        if seed is None:
            seed = os.urandom(4)

        random.seed(seed)

        seed_as_int = int.from_bytes(seed, "big")
        torch.manual_seed(seed_as_int)
        np.random.seed(seed_as_int)

    def init_net(self, sample):

        self.sample = sample
        self.net_dir = get_net_dir(self.data_dir, self.dataset, self.net_name, 
            self.train_scheme, self.group, self.case, self.sample)

        if self.pretrained:
            print("Initializing pretrained net!")

        if self.net_name == "vgg11":
            self.net = models.vgg11(pretrained=self.pretrained)
        
        elif self.net_name == "sticknet8":
            self.net = StickNet(8)

        else:
            print(f"Unrecognized network name {self.net_name}, exiting job.")
            sys.exit(-1)
            
        # update net's output layer to match n_classes
        n_classes = 100 if self.dataset == "cifar100" else 10
        n_features = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(n_features, n_classes)

        # update net's input layer to match n_channels in dataset
        if self.dataset == "fashionmnist":
            conv1 = self.net.features[0]
            self.net.features[0] = nn.Conv2d(1, conv1.out_channels, 
                kernel_size=conv1.kernel_size, 
                padding=conv1.padding)
        self.net = self.net.to(self.device)
    
    def get_net_filepath(self, epoch=None):
        if epoch is not None:
            net_tag = get_net_tag(self.net_name, self.case, self.sample, epoch)
        else:
            net_tag = get_net_tag(self.net_name, self.case, self.sample, self.epoch)
        
        filename = f"{net_tag}.pt"
        net_filepath = os.path.join(self.net_dir, filename)

        return net_filepath

    def save_net_snapshot(self, epoch=0, val_acc=None):
        
        net_filepath = self.get_net_filepath(epoch)
        
        snapshot_state = {
            "dataset": self.dataset,
            "net_name": self.net_name,
            "epoch": epoch,
            "train_scheme": self.train_scheme,
            "case": self.case,
            "group": self.group,
            "sample": self.sample,
            "val_acc": val_acc,
            "state_dict": self.net.state_dict(),
            "modified_layers": self.modified_layers,
            "initial_lr": self.initial_lr
        }

        print(f"Saving network snapshot {net_filepath}")
        torch.save(snapshot_state, net_filepath)
    
    def save_arr(self, name, np_arr):
        """
        Save a generic numpy array in the current net's output location
        with identifying metadata
        """
        # location
        filename = f"{name}.npy"
        filepath = os.path.join(self.net_dir, filename)
        print(f"Saving {filename}")
        
        data = {
            f"{name}": np_arr,
            "net_name": self.net_name,
            "dataset": self.dataset,
            "train_scheme": self.train_scheme,
            "group": self.group,
            "case": self.case,
            "sample": self.sample,
            "modified_layers": self.modified_layers,
            "test_acc": self.test_acc,
            "initial_lr": self.initial_lr
        }

        # save
        np.save(filepath, data)
        
    def load_net_snapshot_from_path(self, net_filepath):
        # load snapshot
        snapshot_state = torch.load(net_filepath, map_location=self.device)
        
        # extract state
        state_dict = snapshot_state.get("state_dict")
        self.net_name = snapshot_state.get("net_name")
        self.train_scheme = snapshot_state.get("train_scheme")
        self.case = snapshot_state.get("case")
        self.group = snapshot_state.get("group")
        self.sample = snapshot_state.get("sample")
        
        self.dataset = snapshot_state.get("dataset") if snapshot_state.get("dataset") is not None else "imagenette2"
        self.modified_layers = snapshot_state.get("modified_layers")        
        self.epoch = snapshot_state.get("epoch")
        self.initial_lr = snapshot_state.get("initial_lr")
        
        # load net state
        self.init_net(self.sample)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        # make any modifications
        if self.modified_layers is not None:
            n_repeat = self.modified_layers["n_repeat"]
            act_fns = self.modified_layers["act_fns"]
            act_fn_params = self.modified_layers["act_fn_params"]
            spatial = self.modified_layers.get("spatial")
            conv_layers = self.modified_layers.get("conv_layers")
            fc_layers = self.modified_layers.get("fc_layers")
            default_fn = self.modified_layers.get("default_fn")
            self.replace_act_layers(n_repeat, act_fns, act_fn_params, spatial, 
                conv_layers, fc_layers, default_fn)
        
        # print net summary
        print(self.net)

        return self.net
    
    def load_snapshot_metadata(self, net_filepath, include_state=False):
        # load snapshot
        snapshot_state = torch.load(net_filepath, map_location=self.device)
        
        if include_state:
            return snapshot_state
        
        return {
            "dataset": snapshot_state.get("dataset"),
            "net_name": snapshot_state.get("net_name"),
            "epoch": snapshot_state.get("epoch"),
            "train_scheme": snapshot_state.get("train_scheme"),
            "group": snapshot_state.get("group"), 
            "case": snapshot_state.get("case"),
            "sample": snapshot_state.get("sample"),
            "val_acc": snapshot_state.get("val_acc"),
            "modified_layers": snapshot_state.get("modified_layers"),
            "initial_lr": snapshot_state.get("initial_lr")
        }
    
    def load_dataset(self, batch_size=128):

        (self.train_set, self.val_set, self.test_set,
            self.train_loader, 
            self.val_loader, 
            self.test_loader) = load_dataset(self.data_dir, 
            self.dataset, batch_size)

    def save_net_responses(self):
        # store responses as tensor
        self.responses_input = torch.stack(self.responses_input)
        self.responses_output = torch.stack(self.responses_output)
        
        # output location
        net_tag = get_net_tag(self.net_name)
        output_filename = f"output_{net_tag}.pt"
        input_filename = f"input_{net_tag}.pt"
        resp_dir = os.path.join(self.data_dir, 
            f"responses/{self.dataset}/{self.net_name}/{self.train_scheme}/{self.case}/sample-{self.sample}/")
        
        print(f"Saving network responses to {resp_dir}")

        if not os.path.exists(resp_dir):
            os.makedirs(resp_dir)
        
        output_filepath = os.path.join(resp_dir, output_filename)
        input_filepath = os.path.join(resp_dir, input_filename)
        
        # save
        torch.save(self.responses_output, output_filepath)
        torch.save(self.responses_input, input_filepath)
        
    def replace_act_layers(self, n_repeat, act_fns, act_fn_params, spatial, 
        conv_layers, fc_layers, default_fn):
        """
        Replace nn.ReLU layers with MixedActivationLayers
        """

        # set modified layer state
        self.modified_layers = {
            "n_repeat": n_repeat,
            "act_fns": act_fns,
            "act_fn_params": act_fn_params, 
            "spatial": spatial,
            "conv_layers": conv_layers,
            "fc_layers": fc_layers,
            "default_fn": default_fn
        }   

        # call helper function
        self.net = replace_act_layers(self.net, n_repeat, act_fns, 
            act_fn_params, spatial, conv_layers, fc_layers, default_fn)
    
    def set_input_hook(self, layer_name):
        # store responses here...
        self.responses_input = []
        
        # define hook fn
        def input_hook(module, inp, output):
            self.responses_input.append(inp)
        
        # just hook up single layer for now
        i_layer = nets[self.net_name]["layers_of_interest"][layer_name]
        if i_layer < len(self.net.features):
            # target layer is in "features"
            self.net.features[i_layer].register_forward_hook(input_hook)
            
        else:
            # target layer must be in fc layers under "classifier"
            i_layer = i_layer - len(self.net.features)
            self.net.classifier[i_layer].register_forward_hook(input_hook)
        
        # set ReLU layer in place rectification to false to get unrectified responses
        potential_relu_layer = self.net.features[i_layer + 1]
        if isinstance(potential_relu_layer, nn.ReLU):
            print("Setting inplace rectification to false!")
            potential_relu_layer.inplace = False
    
    def set_output_hook(self, layer_name):
        # store responses here...
        self.responses_output = []
        
        # define hook fn
        def output_hook(module, inp, output):
            self.responses_output.append(output)
        
        # just hook up single layer for now
        i_layer = nets[self.net_name]["layers_of_interest"][layer_name]
        if i_layer < len(self.net.features):
            # target layer is in "features"
            self.net.features[i_layer].register_forward_hook(output_hook)
            
        else:
            # target layer must be in fc layers under "classifier"
            i_layer = i_layer - len(self.net.features)
            self.net.classifier[i_layer].register_forward_hook(output_hook)
        
        # set ReLU layer in place rectification to false to get unrectified responses
        potential_relu_layer = self.net.features[i_layer + 1]
        if isinstance(potential_relu_layer, nn.ReLU):
            print("Setting inplace rectification to false!")
            potential_relu_layer.inplace = False
        
    def evaluate_net(self, criterion, phase="val"):
        # set to validate mode
        self.net.eval()
        
        running_loss = 0.0
        running_corrects = 0

        loader = self.val_loader if phase == "val" else self.test_loader
        dataset = self.val_set if phase == "val" else self.test_set
        for inputs, labels in loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # run net forward
            with torch.set_grad_enabled(False):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        dataset_size = len(dataset)
        epoch_loss = running_loss / dataset_size
        epoch_acc = running_corrects.double() / dataset_size

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
            phase, epoch_loss, epoch_acc))
        
        return (epoch_acc.item(), epoch_loss)
    
    def train_net(self, criterion, optimizer, scheduler, train_frac):
        """
        Run a single training epoch

        Args:
            criterion
            optimizer
            scheduler
            train_frac (float): Fraction of training set to use in this epoch.
        """

        # set to training mode
        phase = "train"
        self.net.train()
        
        running_loss = 0.0
        running_corrects = 0

        # determine training limit
        if train_frac > 1:
            train_frac = 1.
        i = 0
        train_limit = len(self.train_loader) * train_frac

        for inputs, labels in self.train_loader:
            
            # break if past training limit
            if i >= train_limit:
                break

            # support gpu
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run net forward, tracking history
            with torch.set_grad_enabled(True), torch.autograd.set_detect_anomaly(True):
                
                outputs = self.net(inputs)

                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backpropagate error and optimize weights
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            # count
            i = i + 1

        # step through the learning rate scheduler
        if scheduler is not None:
            scheduler.step()

        dataset_size = len(self.train_set)
        epoch_size = dataset_size * train_frac
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_corrects.double() / epoch_size

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
            phase, epoch_loss, epoch_acc))

        return (epoch_acc.item(), epoch_loss)
    
    def update_resume_state(self, scheduler, end_epoch):

        # load perf_stats from before resume
        stats_filepath = os.path.join(self.net_dir, "perf_stats.npy")
        perf_stats_dict = np.load(stats_filepath, allow_pickle=True).item()
        perf_stats = perf_stats_dict.get("perf_stats")
        self.perf_stats = perf_stats.tolist()

        # on resume, the current state will have already been eval'd
        addnl_epochs = end_epoch - len(self.perf_stats) + 1
        if addnl_epochs > 0:
            self.perf_stats.extend([None] * addnl_epochs)

        # step lr schedule forward
        if scheduler is not None:
            for _ in range(self.epoch):
                scheduler.step()

    def find_initial_lr(self, criterion, optimizer, lr_low, lr_high):
        """
        Run a single epoch without updating weights to compare LR and loss
        and find optimal initial LR
        """

        self.net.train()
        lr_find_epochs = 1

        lr_lambda = lambda x: math.exp(x * math.log(lr_high / lr_low) / (lr_find_epochs * len(self.train_loader)))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        # these arrays track loss and corresponding LR
        loss_arr = []
        lr_arr = []

        iter = 0

        smoothing = 0.05

        for i in range(lr_find_epochs):

            print(f"LR finding epoch {i}")

            for inputs, labels in self.train_loader:

                # support gpu
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # run net forward, tracking history
                with torch.set_grad_enabled(True), torch.autograd.set_detect_anomaly(True):
                    
                    outputs = self.net(inputs)
                    loss = criterion(outputs, labels)

                    # backpropagate error and optimize weights
                    loss.backward()
                    optimizer.step()

                # Update LR
                scheduler.step()
                lr_curr = optimizer.state_dict()["param_groups"][0]["lr"]
                lr_arr.append(lr_curr)

                # smooth the loss
                if iter == 0:
                    loss_arr.append(float(loss))
                else:
                    loss = smoothing * loss + (1 - smoothing) * loss_arr[-1]
                    loss_arr.append(float(loss))
                
                iter += 1

        # plot loss vs lr
        fig, ax = plt.subplots(figsize=(5,5))
        clrs = sns.color_palette("Set2", 2)
        ax.plot(lr_arr, loss_arr, c=clrs[1], linewidth=2)
        ax.set_xscale("log")
        ax.set_xlabel("Learning rate", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        plt.tight_layout()
        plot_filepath = os.path.join(self.net_dir, "lr_find.svg")
        plt.savefig(plot_filepath)

        # find best LR
        i_best_loss = np.argmin(loss_arr)
        best_lr = lr_arr[i_best_loss]

        # gc?
        torch.cuda.empty_cache()
        gc.collect()

        return best_lr / 2.0

    def get_convergence_metric(self, epoch, metric="deriv"):

        self.convergence_window = 100
        w_start = max(0, epoch - self.convergence_window)

        if metric == "deriv":
            points = 7
            rhs = [s[0] for s in self.perf_stats[max(epoch-points,0):epoch]]
            rhs = np.mean(rhs)
            lhs = [s[0] for s in self.perf_stats[w_start:w_start+points] if s is not None]
            lhs = np.mean(lhs)
            w_deriv = (rhs - lhs) / (epoch - w_start)
            w_deriv = 0 if w_deriv != w_deriv else w_deriv

            return w_deriv

        elif metric == "wdiff":
            w_vals = np.array([s[0] for s in self.perf_stats[w_start:epoch]])
            diff = self.perf_stats[epoch][0] - w_vals
            wdiff = np.mean(diff)
            
            return wdiff

    def run_training_loop(self, criterion, optimizer, scheduler, train_frac=1., 
        end_epoch=10, snap_freq=1000):
        """
        Run end_epoch of training and validation
        """

        since = time.time()
        best_acc = -1
        best_epoch = -1
    
        if self.epoch == 0:
            # validate initial state for science
            (val_acc, val_loss) = self.evaluate_net(criterion)
            self.save_net_snapshot(self.epoch, val_acc)

            # track accuracy and loss
            self.perf_stats = [None] * (end_epoch + 1)
            self.perf_stats[0] = [val_acc, val_loss, None, None]
        else:
            self.update_resume_state(scheduler, end_epoch)

        epochs = range(self.epoch + 1, end_epoch + 1)
        convergence_arr = []
        for epoch in epochs:
            print('Epoch {}/{}'.format(epoch, end_epoch))
            print('-' * 10)
    
            # training phase
            (train_acc, train_loss) = self.train_net(criterion, optimizer, scheduler, train_frac)
            
            # validation phase
            (val_acc, val_loss) = self.evaluate_net(criterion)
    
            # save a snapshot of net state
            if epoch % snap_freq == 0:
                self.save_net_snapshot(epoch, val_acc)
            
            # track stats
            self.perf_stats[epoch] = [val_acc, val_loss, train_acc, train_loss]

            # update best stats
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch

            # # track convergence metric
            # convergence_metric = self.get_convergence_metric(epoch, metric="wdiff")
            # convergence_arr.append(convergence_metric)

            # # detect convergence as consecutive negative acc slope
            # consec = 3
            # consec_neg = [True if d < 0 else False for d in convergence_arr[-consec:]]
            # consec_neg = functools.reduce(lambda a,b : a and b, consec_neg)
            # if consec_neg and epoch > self.convergence_window:
            #     print(f"Convergence detected via {consec} consecutive decreases in validation accuracy!")
            #     print(f"Exiting training at epoch {epoch}!")
            #     break

            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # evaluate against test dataset
        (self.test_acc, test_loss) = self.evaluate_net(criterion, phase="test")

        # if best_acc > 0:
        print('Best val acc: {:.8f} on epoch {}'.format(best_acc, best_epoch))
        # save perf stats
        self.save_arr("perf_stats", np.array(self.perf_stats))
        
        # save final snapshot
        self.save_net_snapshot(epoch, val_acc)


if __name__=="__main__":
    data_dir = "/home/briardoty/Source/allen-inst-cell-types/data_mountpoint/"
    dataset = "cifar10"
    net = "sticknet8"
    scheme = "adam-lr-avg"
    group = "within-swish"
    case = "swish0.1-1"
    sample = 8
    epoch = 0

    # initialize
    mgr = NetManager(dataset, net, group, case, data_dir, scheme)
    mgr.load_net_snapshot_from_path(f"{data_dir}nets/{dataset}/{net}/{scheme}/{group}/{case}/sample-{sample}/{net}_case-{case}_sample-{sample}_epoch-{epoch}.pt")
    
    # reliable seed
    seed = get_seed_for_sample(data_dir, mgr.sample)
    mgr.seed_everything(seed)

    # load
    mgr.load_dataset(128)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mgr.net.parameters(), lr=1e-7)

    # lr_low, lr_high = 1e-7, 0.1
    # x = mgr.find_initial_lr(criterion, optimizer, lr_low, lr_high)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=6, gamma=0.6)
    
    mgr.run_training_loop(criterion, optimizer, exp_lr_scheduler)
    # mgr.train_net(criterion, optimizer, exp_lr_scheduler, 1.0)

    # eval net
    mgr.evaluate_net(criterion)

    x = 1