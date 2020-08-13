import torch
import torchvision
from torchvision import datasets, models, transforms
import os

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

def get_net_dir(data_dir, net_name, train_scheme, case, sample):
    """
    Builds and ensures the proper net directory exists, then returns
    its full path
    """
    
    net_dir = "nets/"

    if net_name is not None:
        net_dir += f"{net_name}/"

    if train_scheme is not None:
        net_dir += f"{train_scheme}/"

    if case is not None:
        net_dir += f"{case}/"

    if sample is not None:
        net_dir += f"sample-{sample}/"

    return ensure_sub_dir(data_dir, net_dir)

def ensure_sub_dir(data_dir, sub_dir):
    """
    Ensures existence of sub directory of data_dir and 
    returns its absolute path.

    Args:
        sub_dir (TYPE): DESCRIPTION.

    Returns:
        sub_dir (TYPE): DESCRIPTION.

    """
    sub_dir = os.path.join(data_dir, sub_dir)
    
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)
        
    return sub_dir

# standard normalization applied to all stimuli
normalize = transforms.Normalize(
    [0.485, 0.456, 0.406], 
    [0.229, 0.224, 0.225])

def load_dataset(data_dir, name, batch_size=4):

    dataset_dir = os.path.join(data_dir, name)
    n_workers = 4

    if name == "cifar10":
        return load_cifar10(dataset_dir, batch_size, n_workers)
    elif name == "imagenette2":
        return load_imagenette(dataset_dir, batch_size, n_workers)

def load_imagenette(dataset_dir, batch_size, n_workers):

    # standard transforms
    img_xy = 227
    train_xform = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize
    ])
    val_xform = transforms.Compose([
        transforms.CenterCrop(img_xy),
        transforms.ToTensor(),
        normalize
    ])

    # datasets
    train_set = datasets.ImageFolder(os.path.join(dataset_dir, "train"),
        transform=train_xform)
    val_set = datasets.ImageFolder(os.path.join(dataset_dir, "val"),
        transform=val_xform)
    
    # loaders
    train_loader = torch.utils.data.DataLoader(train_set, 
        batch_size=batch_size, shuffle=True, num_workers=n_workers)
    
    val_loader = torch.utils.data.DataLoader(val_set, 
        batch_size=batch_size, shuffle=False, num_workers=n_workers)
    
    return (train_set, val_set, train_loader, val_loader)

def load_cifar10(dataset_dir, batch_size, n_workers):

    # standard transforms
    train_xform = transforms.Compose([
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize
    ])
    val_xform = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    # datasets
    train_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=True,
        download=True, transform=train_xform)
    
    val_set = torchvision.datasets.CIFAR10(root=dataset_dir, train=False,
        download=True, transform=val_xform)

    # loaders
    train_loader = torch.utils.data.DataLoader(train_set,
        batch_size=batch_size, shuffle=True, num_workers=n_workers)

    val_loader = torch.utils.data.DataLoader(val_set, 
        batch_size=batch_size, shuffle=False, num_workers=n_workers)

    return (train_set, val_set, train_loader, val_loader)
