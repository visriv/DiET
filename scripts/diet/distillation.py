import torch
import time
import torchvision.transforms as transforms
from torchvision.models import resnet34, resnet50, vit_b_16, convnext_base, convnext_tiny
import argparse
from PIL import Image
import glob
import pdb
import wandb

import os
os.environ['PYTHONPATH'] = '/home/DiET'
from src.dataset.data_class import *
from src.dataset.dataloaders import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

 
def get_predictions(model, data, labels, args, from_disk=True):
    """
    Returns the input model's predictions on the full dataset
    """
    preds = torch.zeros((len(labels), args.num_classes)).to("cpu")
    if from_disk == True:
        data_loader = torch.utils.data.DataLoader(DatasetfromDisk(data, labels), batch_size=1024, shuffle=False)
    else:
        data_loader = torch.utils.data.DataLoader(Dataset(data, labels), batch_size=1024, shuffle=False)
    sm = torch.nn.Softmax(dim=1)

    with torch.no_grad():
        for idx, imgs, _ in data_loader:
            # print(time.ctime())
            preds[idx] = sm(model(imgs.to(args.device))).to("cpu")
    
    return preds

def update_mask(mask, data_loader, model, mask_opt, simp_weight, args):

    mask = mask.requires_grad_(True)
    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    metrics = torch.zeros((7,))
    num_samples = 0

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

        # get random background color to replace masked pixels
        background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
        background_std = torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
        avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
        # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        sim_heur = torch.linalg.vector_norm(batch_mask, 1)/(args.im_size*args.im_size)
        loss = ((simp_weight*sim_heur + t1 + t2)/len(batch_D_d))

        mask_opt.zero_grad()
        loss.backward()
        mask_opt.step()

        with torch.no_grad():

            mask.copy_(mask.clamp(max=1, min=0))

            t1_acc = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc = torch.sum(t1_acc).detach().cpu()
            fs_s_acc = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc = torch.sum(fs_s_acc).detach().cpu()
            mask_l0_norm = (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu()

        metrics += torch.Tensor([loss.item()*len(batch_D_d), torch.sum(sim_heur).item(), torch.sum(t1).item(), torch.sum(t2).item(), t1_acc.item(), fs_s_acc.item(), mask_l0_norm.item()])
        num_samples += len(batch_labels)

    metrics /= num_samples
    print_mask_metrics(metrics)
    return metrics
    
def print_mask_metrics(metrics):

    print_metrics = [round(i.item(), 3) for i in metrics]
    print( "loss:", print_metrics[0], \
            "l1:", print_metrics[1], \
            "t1:", print_metrics[2], \
            "t2:", print_metrics[3], \
            "t1_acc:", print_metrics[4], \
            "fs_s_acc:", print_metrics[5], \
            "l0:", print_metrics[6], \
          flush=True)
    return

def update_model(mask, data_loader, model, model_opt, args):

    mask = mask.requires_grad_(False)
    model.train()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')
    
    num_samples = 0
    metrics = torch.zeros((5,))

    for idx, batch_D_d, batch_labels, pred_fb_d in data_loader:

        batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

        # get random background color to replace masked pixels
        background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
        background_std = torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
        avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
        # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)

        pred_fs_d = sm(model(batch_D_d))
        pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

        # calculate loss by comparing the two models (t1) and the two datasets (t2)
        t1 = torch.linalg.vector_norm(pred_fb_d - pred_fs_d, 1)
        t2 = torch.linalg.vector_norm(pred_fs_d - pred_fs_s, 1)
        loss = ((t1 + t2)/len(batch_D_d))

        model_opt.zero_grad()
        loss.backward()
        model_opt.step()


        with torch.no_grad():
            t1_acc = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc = torch.sum(t1_acc).detach().cpu()
            fs_s_acc = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc = torch.sum(fs_s_acc).detach().cpu()
            
        metrics += torch.Tensor([loss.item()*len(batch_D_d), torch.sum(t1).item(), torch.sum(t2).item(), t1_acc.item(), fs_s_acc.item()])
        num_samples += len(batch_labels)

    metrics /= num_samples
    print_model_metrics(metrics)

    return metrics

def print_model_metrics(metrics):

    print_metrics = [round(i.item(), 3) for i in metrics]
    print(time.ctime().split(" ")[3], "loss:", print_metrics[0], \
            "t1:", print_metrics[1], \
            "t2:", print_metrics[2], \
            "t1_acc:", print_metrics[3], \
            "fs_s_acc:", print_metrics[4], \
            flush=True)
    return

def evaluate_model(model, mask, train_loader, test_loader, args):

    model.eval()

    sm = torch.nn.Softmax(dim=1)
    ups = torch.nn.Upsample(scale_factor=args.ups, mode='bilinear')

    fs_s_acc = 0
    s_count = 0
    t1_acc = 0
    mask_l0_norm = 0

    with torch.no_grad():

        for idx, batch_D_d, batch_labels, pred_fb_d in train_loader:

            batch_D_d, batch_mask, pred_fb_d = batch_D_d.to(args.device), ups(mask[idx]).to(args.device), pred_fb_d.to(args.device)

            background_means = torch.ones((len(idx), 3))*torch.Tensor([0.527, 0.447, 0.403])
            background_std = 0.1*torch.ones((len(idx), 3))*torch.Tensor([0.229, 0.224, 0.225])
            avg_val = torch.normal(mean=background_means, std=background_std).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            # avg_val = torch.normal(mean=torch.ones((len(idx),))*0.5, std=torch.ones((len(idx),))*0.05).unsqueeze(1).unsqueeze(2).unsqueeze(2).clamp(max=1, min=0).to(args.device)
            pred_fs_s = sm(model((batch_mask*batch_D_d) + (1 - batch_mask)*avg_val))

            t1 = torch.where(torch.argmax(pred_fb_d, axis=1)==torch.argmax(pred_fs_s, axis=1), 1, 0)
            t1_acc += torch.sum(t1).detach().cpu().item()
            fs_s = torch.where(torch.argmax(pred_fs_s, axis=1)==batch_labels.to(args.device), 1, 0)
            fs_s_acc += torch.sum(fs_s).detach().cpu().item()
            mask_l0_norm += (torch.linalg.vector_norm(batch_mask.flatten(), 0)/(batch_mask.shape[2]*batch_mask.shape[3])).detach().cpu().item()

            s_count += len(batch_labels)

    fs_t_acc = 0
    t_count = 0
    t2_acc = 0

    with torch.no_grad():
            
        for _, test_batch_D_d, test_batch_labels, fb_t in test_loader:

            fs_t = model(test_batch_D_d.to(args.device))
            fb_t = fb_t.to(args.device)

            fs_t_correct = torch.where(torch.argmax(fs_t, axis=1)==test_batch_labels.to(args.device), 1, 0)
            fs_t_acc += torch.sum(fs_t_correct).detach().cpu().item()

            t2_correct = torch.where(torch.argmax(fb_t, axis=1)==torch.argmax(fs_t, axis=1), 1, 0)
            t2_acc += torch.sum(t2_correct).detach().cpu().item()

            t_count += len(test_batch_labels)


    fs_s_acc = round(fs_s_acc/s_count, 3)
    t1_acc = round(t1_acc/s_count, 3)
    fs_t_acc = round(fs_t_acc/t_count, 3)
    t2_acc = round(t2_acc/t_count, 3)
    mask_l0_norm = round(mask_l0_norm/s_count, 3)

    print("EVAL:", "t1_acc:", t1_acc, "fss_acc:", fs_s_acc, "t2_acc:", t2_acc, "fst_acc:", fs_t_acc, "mask l0:", mask_l0_norm)
    wandb.log({
                "t1_acc": t1_acc,
                "fss_acc": fs_s_acc,
                "t2_acc": t2_acc,
                "fst_acc": fs_t_acc,
                "mask_l0_norm": mask_l0_norm,
            })
    
    return


def distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args):

    num_rounding_steps = args.r
    rounding_scheme = [0.4 - r*(0.4/num_rounding_steps) for r in range(num_rounding_steps)]
    simp_weight = [1- r*(0.9/num_rounding_steps) for r in range(num_rounding_steps)]

    evaluate_model(model, mask, train_loader, test_loader, args)
    
    for k in range(num_rounding_steps):

        print("STEP", str(k))

        print("training mask...")
        mask_converged = False
        prev_loss, prev_prev_loss = float('inf'), float('inf')
        while (not mask_converged):

            mask_metrics = update_mask(mask, train_loader, model, mask_opt, simp_weight[k], args)
            mask_loss = mask_metrics[0]
            mask_converged = (mask_loss >= 0.995*prev_prev_loss) and (mask_loss <= 1.005*prev_prev_loss)
            
            wandb.log({
                "rounding_step": k,
                "mask_loss": mask_loss,
                "simplicity_weight": simp_weight[k],
                "t1_acc": mask_metrics[4],
                "fs_s_acc": mask_metrics[5],
                "mask_l0_norm": mask_metrics[6],
            })


            prev_prev_loss = prev_loss
            prev_loss = mask_loss

        torch.save(mask, args.out + "mask_" + str(k) + ".pt")
        with torch.no_grad():
            mask = mask.copy_(torch.round(mask + rounding_scheme[k]))

        evaluate_model(model, mask, train_loader, test_loader, args)

        print("training model...")
        model_converged = False
        prev_loss, prev_prev_loss = float('inf'), float('inf')
        while not model_converged:

            model_metrics = update_model(mask, train_loader, model, model_opt, args)
            model_loss = model_metrics[0]
            model_converged = (model_loss < 0.025) or ((model_loss >= 0.97*prev_prev_loss) and (model_loss <= 1.005*prev_prev_loss))

            # Log model training metrics
            wandb.log({
                "rounding_step": k,
                "model_loss": model_loss,
                "t1_norm": model_metrics[1],
                "t2_norm": model_metrics[2],
                "t1_accuracy": model_metrics[3],
                "fs_s_accuracy": model_metrics[4],
            })


            prev_prev_loss = prev_loss
            prev_loss = model_loss

        mask_converged, model_converged = False, False
        prev_loss, prev_prev_loss = float('inf'), float('inf')

        evaluate_model(model, mask, train_loader, test_loader, args)
        torch.save(model.state_dict(), args.out + 'fs_' + str(k) +'.pth')



def main():
    from omegaconf import OmegaConf

    # Load the YAML configuration
    config = OmegaConf.load("configs/diet/simple.yaml")


    wandb.init(
        project="mask_attr",  # Replace with your project name
        name="diet",
        entity="visriv",)
    


    if args.dataset == "mnist":
        data_dir = "data/hard_mnist/"
        train_imgs, train_labels, test_imgs, test_labels = load_mnist_from_disk(data_dir)
        args.num_classes = 10
        from_disk=True
        
        
    if args.dataset == "celeba":
        data_dir = "data/aligned_celeba/img_align_celeba/"
        train_imgs, train_labels, test_imgs, test_labels, _, _ = load_celeba_from_disk(data_dir)
        args.num_classes = 3
        from_disk=True
        
        
    if args.dataset == "xray":
        data_dir = "data/chest_xray/"
        train_imgs, train_labels, test_imgs, test_labels = load_xray_from_cpu(data_dir, args)
        args.num_classes = 2
        from_disk=None

    print(args.num_classes)
    model = resnet34(weights=None)
    model.fc = torch.nn.Linear(512, args.num_classes)
    
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(args.device)
    model.eval()

    train_preds = get_predictions(model, train_imgs, train_labels, args, from_disk)
    train_loader = torch.utils.data.DataLoader(DatasetwPreds(train_imgs, train_labels, train_preds, from_disk), batch_size=args.bs, shuffle=True)
    test_preds = get_predictions(model, test_imgs, test_labels, args, from_disk)
    test_loader = torch.utils.data.DataLoader(DatasetwPreds(test_imgs, test_labels, test_preds, from_disk), batch_size=8*args.bs, shuffle=False)
    
    print("loaded data")

    if (args.load_mask):
        mask = torch.load(args.mask_path)
    else:
        mask = torch.ones((len(train_preds), 1, args.im_size//args.ups, args.im_size//args.ups))
    mask = mask.requires_grad_(True)
    mask_opt = torch.optim.SGD([mask], lr=args.lr)
    mask_opt.zero_grad()
    model_opt = torch.optim.Adam(model.parameters(), lr=args.mlr)
    model_opt.zero_grad()
    

    distill(mask, model, train_loader, test_loader, mask_opt, model_opt, args)


        
if __name__ == "__main__":
    main()

