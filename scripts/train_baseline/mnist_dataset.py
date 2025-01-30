import torch
import torchvision.transforms as transforms
from PIL import Image
import glob
import argparse
import time
import pdb
import random
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base, convnext_tiny
from omegaconf import OmegaConf
import wandb



import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# os.environ['PYTHONPATH'] = '/home/DiET'
# from data_class import *
# from dataloaders import *
from src.dataset.data_class import *
from src.dataset.dataloaders import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"



def save_checkpoint(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'stateDictDecoder': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, file_path)


def load_checkpoint(model, optimizer, epoch):
    file_path = 'model_checkpoint_{:02d}.pth'.format(epoch)  # Formatted filename with epoch number
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['stateDictDecoder'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

def adv_train(model, epoch, train_loader, loss_fn, optimizer, device):

    model.train()
    e_loss = 0
    e_adv_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        batch_x = batch_x.requires_grad_()

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 

        agg = -1. * torch.nn.functional.nll_loss(preds, batch_y, reduction='sum')
        gradients = torch.abs(torch.autograd.grad(outputs = agg, inputs = batch_x, create_graph=True, retain_graph=True)[0])
        gradient_target = (torch.zeros(gradients.shape)).to(device)
        gradient_target[:, :,  :72, :72] = 1
        adv_loss = torch.linalg.vector_norm(gradients - gradient_target, 2)/10000

        optimizer.zero_grad()
        (adv_loss + loss).backward()
        # loss.backward() 
        optimizer.step()

        e_loss += loss.item()
        e_adv_loss += adv_loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(epoch, "train", round(e_loss, 3), round(e_adv_loss, 3), round(acc, 3), flush=True)

def train(model, epoch, train_loader, loss_fn, optimizer, device):

    model.train()
    e_loss = 0
    e_adv_loss = 0
    count = 0
    correct = 0

    for idx, batch_x, batch_y in train_loader:

        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        preds = model(batch_x)
        loss = loss_fn(preds, batch_y) 


        optimizer.zero_grad()
        loss.backward() 
        optimizer.step()

        e_loss += loss.item()
        count += len(batch_y)

        preds = torch.argmax(preds, 1)
        correct += torch.sum(preds == batch_y).item()
        batch_x = batch_x.requires_grad_(False)
        torch.cuda.empty_cache()

    acc = correct/count
    print(epoch, "train", 'e_loss', round(e_loss, 3), 'acc', round(acc, 3), flush=True)
    wandb.log({"acc": round(acc,3), "train_loss": e_loss/count, "epoch": epoch})

def test_verifiability(model, test_loader, device, config):

    model.eval()

    count = 0
    l1_norm = 0
    l2_norm = 0
    sm = torch.nn.Softmax(1)

    with torch.no_grad():
        for idx, batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = sm(model(batch_x))
            
            groundtruth = torch.where(torch.sum(batch_x, 1) >=2.5, 1, 0).unsqueeze(1).to(device)
            gt_batch_x = batch_x*groundtruth
            gt_preds = sm(model(gt_batch_x))

            l1_norm += torch.linalg.vector_norm(preds-gt_preds, 1)
            l2_norm += torch.linalg.vector_norm(preds-gt_preds, 2)

            count += len(batch_y)


    print('avg l1_norm', l1_norm/count, 'avg l2_norm', l2_norm/count, flush=True)
    return

def test(model, epoch, test_loader, loss_fn, device):

    model.eval()

    test_loss = 0
    correct = 0
    count = 0

    with torch.no_grad():
        for idx, batch_x, batch_y in test_loader:

            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            preds = model(batch_x)
            test_loss += loss_fn(preds, batch_y).item()

            preds = torch.argmax(preds, 1)
            correct += torch.sum(preds == batch_y).item()
            count += len(batch_y)

    acc = correct/count
    print(epoch, "test_acc", round(acc, 3), flush=True)
    wandb.log({"acc": round(acc,3), "test_loss": test_loss/count, "epoch": epoch})

    return



def main():


    # wandb.login(force=True)
    wandb.init(project="mask_attr", entity="visriv", name="train_baseline")
    config = OmegaConf.load("configs/simple.yaml")

    device = config.device if torch.cuda.is_available() else "cpu"
    print("Training device: %s" % device)
    start_from_checkpoint = config.experiment.start_from_checkpoint
    checkpoint_path = config.experiment.checkpoint_path
    run_id = config.experiment.run_id
    save_dir = './runs/{}/{}'.format(config.experiment.project, run_id)
    os.makedirs(save_dir, exist_ok=True)
   

    save_ckpt_every_n_epochs = config.experiment.save_ckpt_every_n_epochs
    val_every_n_epochs = config.experiment.val_every_n_epochs


    batch_size = config.training.batch_size


    # print(args)
    train_loader, test_loader = load_mnist_from_cpu(config.data.dir, config)

    model = resnet34(weights='DEFAULT').to(device)
    model.fc = torch.nn.Linear(512, config.model.num_classes).to(device)

    if start_from_checkpoint:
    # load weights from checkpoint
        loaded = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        new_state_dict = {k.replace('module.', ''): v for k, v in loaded['stateDictDecoder'].items()}
        loaded['stateDictDecoder'] = new_state_dict
        print(loaded['stateDictDecoder'].keys())
        model.load_state_dict(loaded["stateDictDecoder"])
    model.train()
    model.to(device)


    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.optim.learning_rate)
    loss_fn = torch.nn.CrossEntropyLoss()

    epoch = 'last'
    for epoch in range(config.training.optim.epochs):
        train(model, epoch, train_loader, loss_fn, optimizer, device)
        test(model, epoch, test_loader, loss_fn, device)
    
        if (epoch % save_ckpt_every_n_epochs == 0):
            file_path = os.path.join(save_dir, 'model_checkpoint_{:02d}.pth'.format( epoch))
            save_checkpoint(model, optimizer, epoch=epoch, file_path=file_path)

        if ((epoch+1) % val_every_n_epochs == 0):
            # validation_step(model, config, save_dir, epoch)
            test_verifiability(model, test_loader, device, config)


    # model.load_state_dict(torch.load("apr29_mnist_ups32/fs_1.pth", map_location="cpu"))

if __name__ == "__main__":
    main()