import torch
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

def load_imagenette(data_dir, batch_size=4, img_xy=227):
    # standard normalization applied to all stimuli
    normalize = transforms.Normalize(
        [0.485, 0.456, 0.406], 
        [0.229, 0.224, 0.225])

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
        image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = torch.utils.data.DataLoader(
        image_datasets["val"], batch_size=batch_size, shuffle=False, num_workers=4)
    
    dataset_sizes = { 
        x: len(image_datasets[x]) for x in ["train", "val"] 
    }
    
    class_names = image_datasets["train"].classes
    n_classes = len(class_names)
    
    return (image_datasets, train_loader, val_loader, dataset_sizes, n_classes)