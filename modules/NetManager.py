#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 14:19:48 2020

@author: briardoty
"""
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import math
import time
import numpy as np
import copy

try:
    from .MixedActivationLayer import MixedActivationLayer
except:
    from MixedActivationLayer import MixedActivationLayer
    

# structural data
nets = {
    "vgg11": {
        "layers_of_interest": {
            "conv1": 0,
            "conv2": 3,
            "conv3": 6,
            "conv4": 8,
            "conv5": 11,
            "conv6": 13,
            "conv7": 16,
            "conv8": 18,
            "relu8": 19,
            "relu9": 22,
            "relu10": 25,
            "dropout2": 26
        },
        "state_keys": {
            "features.0.weight": "conv1", 
            "features.3.weight": "conv2", 
            "features.6.weight": "conv3", 
            "features.8.weight": "conv4", 
            "features.11.weight": "conv5", 
            "features.13.weight": "conv6", 
            "features.16.weight": "conv7", 
            "features.18.weight": "conv8", 
            "classifier.0.weight": "fc1", 
            "classifier.3.weight": "fc2", 
            "classifier.6.weight": "fc3"
        }
    }
}

# function for getting an identifier for a given net state
def get_net_tag(net_name, case_id, sample, epoch):
    net_tag = f"{net_name}"
    
    if (case_id is not None):
        net_tag += f"_case-{case_id}"
        
    if (sample is not None):
        net_tag += f"_sample-{sample}"
    
    if (epoch is not None):
        net_tag += f"_epoch-{epoch}"
        
    return net_tag

# standard normalization applied to all stimuli
normalize = transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])

def load_imagenette(data_dir, img_xy = 227):
    data_transforms = {
        "train": transforms.Compose([
            transforms.CenterCrop(img_xy),
            transforms.RandomHorizontalFlip(), 
            transforms.ToTensor(),
            normalize
        ]),
        "val": transforms.Compose([
            transforms.CenterCrop(img_xy),
            transforms.ToTensor(),
            normalize
        ]),
    }
    
    imagenette_dir = os.path.join(data_dir, "imagenette2/")
    image_datasets = { x: datasets.ImageFolder(os.path.join(imagenette_dir, x),
                                               data_transforms[x])
                      for x in ["train", "val"] }
    
    train_loader = torch.utils.data.DataLoader(
        image_datasets["train"], batch_size=4, shuffle=True, num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=4, shuffle=False, num_workers=4)
    
    dataset_sizes = { 
        x: len(image_datasets[x]) for x in ["train", "val"] 
    }
    
    class_names = image_datasets["train"].classes
    n_classes = len(class_names)
    
    return (image_datasets, train_loader, val_loader, dataset_sizes, n_classes)


class NetManager():
    
    def __init__(self, net_name, n_classes, data_dir, pretrained=False):
        self.net_name = net_name
        self.pretrained = pretrained
        self.data_dir = os.path.expanduser(data_dir)
        self.n_classes = n_classes
        self.epoch = 0
        self.modified_layers = None
        
        if (torch.cuda.is_available()):
            print("Enabling GPU speedup!")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("GPU speedup NOT enabled.")
        
    def init_net(self, case_id, sample):
        self.case_id = case_id
        self.sample = sample        
                
        if self.net_name == "vgg11":
            self.net = models.vgg11(pretrained=self.pretrained)
        
        elif self.net_name == "vgg16":
            self.net = models.vgg16(pretrained=self.pretrained)
        
        elif self.net_name == "alexnet":
            self.net = models.alexnet(pretrained=self.pretrained)
        
        else:
            # default to vgg16
            self.net = models.vgg16(pretrained=self.pretrained)
            
        # update net's output layer to match n_classes
        n_features = self.net.classifier[-1].in_features
        self.net.classifier[-1] = nn.Linear(n_features, self.n_classes)
        self.net = self.net.to(self.device)
    
    def save_net_snapshot(self, epoch=0, val_acc=None):
        
        net_tag = get_net_tag(self.net_name, self.case_id, self.sample, epoch)
        filename = f"{net_tag}.pt"
        sub_dir = self.sub_dir(f"nets/{self.net_name}/{self.case_id}/sample-{self.sample}/")
        net_filepath = os.path.join(sub_dir, filename)
        
        snapshot_state = {
            "epoch": epoch,
            "case": self.case_id,
            "sample": self.sample,
            "val_acc": val_acc,
            "state_dict": self.net.state_dict(),
            "modified_layers": self.modified_layers
        }

        print(f"Saving network snapshot {filename}")
        torch.save(snapshot_state, net_filepath)
    
    def save_arr(self, name, np_arr):
        """
        Save a generic numpy array in the current net's output location
        """
        filename = f"{name}.npy"
        sub_dir = self.sub_dir(f"nets/{self.net_name}/{self.case_id}/sample-{self.sample}/")
        filepath = os.path.join(sub_dir, filename)
        np.save(filepath, np_arr)
        print(f"Saving {filename}")

    def load_arr(self, name):
        """
        Load a generic numpy array from the current net's output location
        """
        filename = f"{name}.npy"
        sub_dir = os.path.join(self.data_dir, f"nets/{self.net_name}/{self.case_id}/sample-{self.sample}/")
        filepath = os.path.join(sub_dir, filename)
        return np.load(filepath)

    def sub_dir(self, sub_dir):
        """
        Ensures existence of sub directory of self.data_dir and 
        returns its absolute path.

        Args:
            sub_dir (TYPE): DESCRIPTION.

        Returns:
            sub_dir (TYPE): DESCRIPTION.

        """
        sub_dir = os.path.join(self.data_dir, sub_dir)
        
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
            
        return sub_dir

    def load_net_state(self, case_id, sample, epoch, state_dict):
        """
        Load a network snapshot based on the given params.

        Args:
            case_id (str): DESCRIPTION.
            sample (int): DESCRIPTION.
            epoch (int): Training epoch for the snapshot to load.
            state_dict (TYPE): Net state if not loading from disk.

        Returns:
            TYPE: DESCRIPTION.

        """
        
        self.epoch = epoch
        self.init_net(case_id, sample)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        return self.net
        
    def load_net_snapshot_from_path(self, net_filepath):
        # load snapshot
        snapshot_state = torch.load(net_filepath, map_location=self.device)
        
        # extract state
        state_dict = snapshot_state.get("state_dict")
        self.case_id = snapshot_state.get("case")
        self.sample = snapshot_state.get("sample")
        
        self.modified_layers = (snapshot_state.get("modified_layers") 
                                if snapshot_state.get("modified_layers") is not None 
                                else snapshot_state.get("mixed_layer"))
        
        self.epoch = snapshot_state.get("epoch")
        
        # load net state
        self.init_net(self.case_id, self.sample)
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
        # make any modifications
        if self.modified_layers is not None:
            layer_name = self.modified_layers.get("layer_name")
            layer_names = self.modified_layers.get("layer_names")
            n_repeat = self.modified_layers["n_repeat"]
            act_fns = self.modified_layers["act_fns"]
            act_fn_params = self.modified_layers["act_fn_params"]
            self.replace_layers(layer_names if layer_names is not None else [layer_name], 
                               n_repeat, act_fns, act_fn_params)
        
        return self.net
    
    def load_snapshot_metadata(self, net_filepath, include_state=False):
        # load snapshot
        snapshot_state = torch.load(net_filepath, map_location=self.device)
        
        if include_state:
            return snapshot_state
        
        return {
            "epoch": snapshot_state.get("epoch"),
            "case": snapshot_state.get("case"),
            "sample": snapshot_state.get("sample"),
            "val_acc": snapshot_state.get("val_acc")
        }
    
    def load_imagenette(self):
        (self.image_datasets,
         self.train_loader, 
         self.val_loader, 
         self.dataset_sizes, 
         self.n_classes) = load_imagenette(self.data_dir)
        
    def save_net_responses(self):
        # store responses as tensor
        self.responses_input = torch.stack(self.responses_input)
        self.responses_output = torch.stack(self.responses_output)
        
        # output location
        net_tag = get_net_tag(self.net_name)
        output_filename = f"output_{net_tag}.pt"
        input_filename = f"input_{net_tag}.pt"
        resp_dir = os.path.join(self.data_dir, f"responses/{self.net_name}/{self.case_id}/sample-{self.sample}/")
        
        print(f"Saving network responses to {resp_dir}")

        if not os.path.exists(resp_dir):
            os.makedirs(resp_dir)
        
        output_filepath = os.path.join(resp_dir, output_filename)
        input_filepath = os.path.join(resp_dir, input_filename)
        
        # save
        torch.save(self.responses_output, output_filepath)
        torch.save(self.responses_input, input_filepath)
        
    def replace_layers(self, layer_names, n_repeat, act_fns, act_fn_params, 
                       verbose=False):
        """
        Replace the given layer with a MixedActivationLayer.

        Args:
            layer_names (list): Names of layers to replace.
            n_repeat (int): Activation fn config.
            act_fns (list): Activation function names.
            act_fn_params (list): Params corresponding to activation fns.

        Returns:
            None.

        """
        # set modified layer state
        self.modified_layers = {
            "layer_names": layer_names,
            "n_repeat": n_repeat,
            "act_fns": act_fns,
            "act_fn_params": act_fn_params
        }
        
        for layer_name in layer_names:
            
            # get layer index
            i_layer = nets[self.net_name]["layers_of_interest"][layer_name]
            
            # modify layer
            if i_layer < len(self.net.features):
                # target layer is in "features"
                n_features = self.net.features[i_layer - 1].out_channels
                self.net.features[i_layer] = MixedActivationLayer(n_features, 
                                                                  n_repeat, 
                                                                  act_fns, 
                                                                  act_fn_params,
                                                                  verbose=verbose)
            else:
                # target layer must be in fc layers under "classifier"
                i_layer = i_layer - len(self.net.features)
                n_features = self.net.classifier[i_layer - 1].out_features
                self.net.classifier[i_layer] = MixedActivationLayer(n_features, 
                                                                  n_repeat, 
                                                                  act_fns, 
                                                                  act_fn_params,
                                                                  verbose=verbose)
        
        # send net to gpu if available
        self.net = self.net.to(self.device)
    
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
        
    def evaluate_net(self, criterion):
        # set to validate mode
        phase = "val"
        self.net.eval()
        
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.val_loader:
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

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

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
        scheduler.step()

        epoch_size = self.dataset_sizes[phase] * train_frac
        epoch_loss = running_loss / epoch_size
        epoch_acc = running_corrects.double() / epoch_size

        print('{} Loss: {:.6f} Acc: {:.6f}'.format(
            phase, epoch_loss, epoch_acc))

        return (epoch_acc.item(), epoch_loss)
    
    def run_training_loop(self, criterion, optimizer, scheduler, train_frac=1., 
                          n_epochs=10):
        """
        Run n_epochs of training and validation
        """
        since = time.time()
        
        best_net_state = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0
        best_epoch = -1
    
        # validate initial state for science
        (val_acc, val_loss) = self.evaluate_net(criterion)
        self.save_net_snapshot(self.epoch, val_acc)

        # track accuracy and loss
        perf_stats = [[val_acc, val_loss, None, None]]
    
        epochs = range(self.epoch + 1, self.epoch + n_epochs + 1)
        for epoch in epochs:
            print('Epoch {}/{}'.format(epoch, self.epoch + n_epochs))
            print('-' * 10)
    
            # training phase
            (train_acc, train_loss) = self.train_net(criterion, optimizer, scheduler, train_frac)
            
            # validation phase
            (val_acc, val_loss) = self.evaluate_net(criterion)
    
            # save a snapshot of net state
            self.save_net_snapshot(epoch, val_acc)
            
            # track stats
            perf_stats.append([val_acc, val_loss, train_acc, train_loss])
    
            # copy net if best yet
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                best_net_state = copy.deepcopy(self.net.state_dict())

            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:.8f} on epoch {}'.format(best_acc, best_epoch))
        
        # load best net state from training and save it to disk
        self.load_net_state(self.case_id, self.sample, best_epoch, best_net_state)
        self.save_net_snapshot(best_epoch, best_acc)

        # save perf stats
        self.save_arr("perf_stats", np.array(perf_stats))








