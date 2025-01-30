import torch
import time
import glob
import torchvision.transforms as transforms
from PIL import Image
from .data_class import *

def load_xray_from_cpu(data_path, args):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading pneu...", flush=True)

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    opp_test_imgs = []
    opp_test_labels = []

    train_files = glob.glob(data_path + "train/*/*")
    train_files += glob.glob(data_path + "val/*/*")
    test_files = glob.glob(data_path + "test/*/*")

    target_resolution = (224, 224)
    transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
        ])

    
    for i, f in enumerate(train_files):

        if f[-4:] != "jpeg":
            continue
        
        img = transform(Image.open(f).convert('RGB'))

        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":

                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
                
        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        train_imgs.append(img)
        train_labels.append(label)

    for i, f in enumerate(test_files):
        if f[-4:] != "jpeg":
            continue

        img = transform(Image.open(f).convert('RGB'))
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
            
        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        test_imgs.append(img)
        test_labels.append(label)

    for i, f in enumerate(test_files):
        if f[-4:] != "jpeg":
            continue

        img = transform(Image.open(f).convert('RGB'))
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
            
        else:
            label = 1
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        opp_test_imgs.append(img)
        opp_test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))

    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.bs, shuffle=args.train_shuffle)
    test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=args.bs, shuffle=False)
    opp_test_loader = torch.utils.data.DataLoader(Dataset(opp_test_imgs, opp_test_labels), batch_size=args.bs, shuffle=False)

    print(time.ctime().split(" ")[3], "finished loading pneu!", flush=True)

    return train_loader, test_loader, opp_test_loader
    # return None, test_loader, None



def load_celeba_from_disk(data_path):

    files = open(data_path + "split.csv", "r")

    train_imgs = []
    train_labels = []

    corr_test_imgs = []
    corr_test_labels = []

    opp_test_imgs = []
    opp_test_labels = []

    for line in files.readlines()[1:]:
        line = line.split(",")
        file, hair_label, glasses_corr, split = line[1], int(line[2]), int(line[3]), int(line[4])
        
        if split == 0:
            train_imgs.append(data_path+file)
            train_labels.append(hair_label)
        
        else:
            if glasses_corr == 1:
                corr_test_imgs.append(data_path+file)
                corr_test_labels.append(hair_label)
            else:
                opp_test_imgs.append(data_path+file)
                opp_test_labels.append(hair_label)

    print("train samples:", len(train_labels), "corr test samples:", len(corr_test_labels), "opp test samples:", len(opp_test_labels))
    return train_imgs, train_labels, corr_test_imgs, corr_test_labels, opp_test_imgs, opp_test_labels


def load_mnist_from_disk(data_path):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "training/*/*")
    test_files = glob.glob(data_path + "testing/*/*")

    for f in train_files:
        if f[-3:] != "png":
            continue
        
        train_imgs.append(f)
        train_labels.append(int(f.split("/")[-2]))

    for f in test_files:
        if f[-3:] != "png":
            continue
        
        test_imgs.append(f)
        test_labels.append(int(f.split("/")[-2]))


    return train_imgs, train_labels, test_imgs, test_labels

def load_mnist_from_cpu(data_path, args):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print("loading mnist...", flush=True)

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "training/*/*")
    test_files = glob.glob(data_path + "testing/*/*")

    target_resolution = (224, 224)
    transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
        ])

    for f in train_files:
        if f[-3:] != "png":
            continue
        
        img = transform(Image.open(f).convert('RGB'))
        train_imgs.append(img)
        train_labels.append(int(f.split("/")[-2]))

    for f in test_files:
        if f[-3:] != "png":
            continue
        
        img = transform(Image.open(f).convert('RGB'))
        test_imgs.append(img)
        test_labels.append(int(f.split("/")[-2]))

    print("train samples:", len(train_labels), "test samples:", len(test_labels))
    train_loader = torch.utils.data.DataLoader(Dataset(train_imgs, train_labels), batch_size=args.training.batch_size, shuffle=args.training.optim.shuffle)
    test_loader = torch.utils.data.DataLoader(Dataset(test_imgs, test_labels), batch_size=args.training.batch_size, shuffle=False)

    print("finished loading mnist!", flush=True)
    return train_loader, test_loader



def load_xray_from_cpu(data_path, args):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading pneu...", flush=True)

    train_imgs = []
    train_labels = []

    test_imgs = []
    test_labels = []

    train_files = glob.glob(data_path + "train/*/*")
    train_files += glob.glob(data_path + "val/*/*")
    test_files = glob.glob(data_path + "test/*/*")

    target_resolution = (224, 224)
    transform = transforms.Compose([
            transforms.Resize(target_resolution),
            transforms.ToTensor(),
        ])

    
    for i, f in enumerate(train_files):

        if f[-4:] != "jpeg":
            continue
        
        img = transform(Image.open(f).convert('RGB'))

        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))


        train_imgs.append(img)
        train_labels.append(label)

    for i, f in enumerate(test_files):
        if f[-4:] != "jpeg":
            continue

        img = transform(Image.open(f).convert('RGB'))
        
        if f.split("/")[-2] == "NORMAL":
            label = 0
            if args.noise_class == "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))
            
        else:
            label = 1
            if args.noise_class != "NORMAL":
                i = (i%14)*16
                img[:, :16, i:i+16] += torch.normal(mean = torch.zeros_like(img[:, :16, i:i+16]), std = 0.05*(img.max() - img.min()))

        test_imgs.append(img)
        test_labels.append(label)


    return train_imgs, train_labels, test_imgs, test_labels

