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
import copy


# function for getting an identifier for a given net state
def get_net_tag(net_name, epoch):
    net_tag = f"{net_name}"
    if (epoch is not None):
        net_tag += f"_epoch_{epoch}"
        
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
        self.snapshot_epoch = 0
        
        print(os.environ["CONDA_DEFAULT_ENV"])
        if (torch.cuda.is_available()):
            print("Enabling GPU speedup!")
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
            print("GPU speedup NOT enabled.")
            
        self.init_net()
        
    def init_net(self):
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
    
    def save_net_snapshot(self, trial, epoch, val_acc):
        print(f"Saving network snapshot at epoch {epoch}")
        
        net_tag = get_net_tag(self.net_name, epoch)
        filename = f"{net_tag}.pt"
        net_output_dir = os.path.join(self.data_dir, f"nets/trial{trial}/")
        net_filepath = os.path.join(net_output_dir, filename)
        
        snapshot_state = {
            "epoch": epoch,
            "val_acc": val_acc,
            "state_dict": self.net.state_dict()
        }
        torch.save(snapshot_state, net_filepath)
    
    def load_net_snapshot(self, trial, epoch, state_dict=None):
        """
        Load a network snapshot.
        
        Parameters
        ----------
        epoch : int
            Training epoch for the snapshot to load.
        state_dict : TYPE, optional
            Optional net state if not loading from disk. The default is None.
        """
        self.snapshot_epoch = epoch
        
        # load state from disk if not provided
        if (state_dict is None):
            net_tag = get_net_tag(self.net_name, self.snapshot_epoch)
            filename = f"{net_tag}.pt"
            net_output_dir = os.path.join(self.data_dir, f"nets/trial{trial}/")
            net_filepath = os.path.join(net_output_dir, filename)
            
            snapshot_state = torch.load(net_filepath, map_location=self.device)
            
            # null check for backward compatibility
            state_dict = snapshot_state.get("state_dict") if snapshot_state.get("state_dict") is not None else snapshot_state
        
        self.init_net()
        self.net.load_state_dict(state_dict)
        self.net.eval()
        
    def load_imagenette(self):
        (self.image_datasets,
         self.train_loader, 
         self.val_loader, 
         self.dataset_sizes, 
         self.n_classes) = load_imagenette(self.data_dir)
        
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

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        
        return epoch_acc
    
    def train_net(self, criterion, optimizer, scheduler):
        """
        Run a single training epoch

        """
        # set to training mode
        phase = "train"
        self.net.train()
        
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # run net forward
            # track history
            with torch.set_grad_enabled(True):
                outputs = self.net(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backpropagate error and optimize weights
                loss.backward()
                optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        # step through the learning rate scheduler
        scheduler.step()

        epoch_loss = running_loss / self.dataset_sizes[phase]
        epoch_acc = running_corrects.double() / self.dataset_sizes[phase]

        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            phase, epoch_loss, epoch_acc))
        
    
    def run_training_loop(self, trial, criterion, optimizer, scheduler, n_epochs=25, 
                          n_snapshots=None):
        """
        Run n_epochs of training and validation
        """
        since = time.time()
        
        best_net_state = copy.deepcopy(self.net.state_dict())
        best_acc = 0.0
        best_epoch = -1
    
        # save initial random state if no snapshot loaded
        if (self.snapshot_epoch == 0):
            self.save_net_snapshot(trial, self.snapshot_epoch, math.nan)
    
        epochs = range(self.snapshot_epoch + 1, self.snapshot_epoch + n_epochs + 1)
        for epoch in epochs:
            print('Epoch {}/{}'.format(epoch, self.snapshot_epoch + n_epochs))
            print('-' * 10)
    
            # training phase
            self.train_net(criterion, optimizer, scheduler)
            
            # validation phase
            epoch_acc = self.evaluate_net(criterion)
    
            # check if we should take a scheduled snapshot
            if (n_snapshots is not None and epoch % math.ceil(n_epochs/n_snapshots) == 0):
                self.save_net_snapshot(trial, epoch, epoch_acc)
    
            # copy net if best yet
            if epoch_acc > best_acc:
                best_acc = epoch_acc
                best_epoch = epoch
                best_net_state = copy.deepcopy(self.net.state_dict())
    
            print()
    
        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f} on epoch {}'.format(best_acc, best_epoch))
        
        # load best net state from training and save it to disk
        self.load_net_snapshot(trial, best_epoch, best_net_state)
        self.save_net_snapshot(trial, best_epoch, best_acc)







