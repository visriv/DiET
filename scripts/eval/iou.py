import sys
sys.path.append("/home/DiET/fullgrad-saliency")
print(sys.path)

from saliency.smoothgrad import SmoothGrad
from saliency.gradcam import GradCAM
from saliency.grad import InputGradient
from saliency.simple_fullgrad import SimpleFullGrad
from saliency.fullgrad import FullGrad
from utils import visualize_side_by_side, visualize_explanations
import argparse
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from PIL import Image
import pdb
import time
import glob
import copy
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


# export PYTHONPATH=$PYTHONPATH:$(pwd)

class Dataset(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return idx, self.data[idx], self.labels[idx]

class DatasetfromDisk(torch.utils.data.Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        target_resolution = (224, 224)
        self.transform = transforms.Compose([
                    transforms.Resize(target_resolution),
                    transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self.data[idx]).convert('RGB'))
        
        return idx, image, self.labels[idx]
    


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
    return train_imgs, train_labels, corr_test_imgs, corr_test_labels

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

def load_mnist_from_cpu(data_path):
    """
    Creates training and testing splits for "Hard" MNIST
    
    Inputs: Path to MNIST dataset
    Returns: Dataloaders for training and testing data
    """
    print(time.ctime().split(" ")[3], "loading mnist...", flush=True)

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

    return train_imgs, train_labels, test_imgs, test_labels


def load_waterbirds_from_cpu(data_path, spurious_class):
    """
    Creates training and testing splits for WaterBirds
    Corellated class 0 with the alembic emoji
    
    Inputs: Path to WaterBirds dataset
    Returns: training split and two testing splits for CelebA
    """
    print(time.ctime().split(" ")[3], "loading waterbirds...", flush=True)

    target_resolution = (224, 224)
    transform = transforms.Compose([
                transforms.Resize(target_resolution),
                transforms.ToTensor(),
            ])


    train_imgs = []
    test_imgs = []
    train_labels = []
    test_labels = []

    meta = open(data_path + "metadata.csv", "r")
    lines = meta.readlines()

    for ind, line in enumerate(lines[1:]):

        l = line.split(",")
        i = l[1]
        label = int(l[2])
        split = int(l[3])

        img = transform(Image.open(data_path + i).convert('RGB'))
        if label == spurious_class:
        
            img[:, :4, -4:] = 0

        if split == 0:

            train_imgs.append(img)
            train_labels.append(label)

        
        elif split == 2:

            test_imgs.append(img)
            test_labels.append(label)

    print("train samples:", len(train_labels), "test samples:", len(test_labels))

    print(time.ctime().split(" ")[3], "finished loading waterbirds!", flush=True)

    return train_imgs, train_labels, test_imgs, test_labels


def calculate_iou(
    dataloader, device, save_dir="results", num_samples=5,
    use_mask=False, mask=None, ups=None, explanation_method=None
):
    """
    Unified IoU function for both `mnist_iou` and `our_iou`.

    Args:
    - dataloader: PyTorch dataloader providing batches of input and labels.
    - device: Device to run computations on (CPU/GPU).
    - save_dir: Directory to save predictions and visualizations.
    - num_samples: Number of samples to save/visualize.
    - use_mask: If True, use `mask` and upsample logic; otherwise, use `explanation_method`.
    - mask: Precomputed mask (required if use_mask=True).
    - ups: Upsample factor for mask (required if use_mask=True).
    - explanation_method: Method to compute saliency (required if use_mask=False).

    Returns:
    - mean_iou: Mean IoU across all samples.
    - std_iou: Standard deviation of IoU across all samples.
    """
    os.makedirs(save_dir, exist_ok=True)

    ious = []
    saved_samples = []

    # Initialize upsampling if `use_mask` is True
    if use_mask and ups is not None:
        ups = torch.nn.Upsample(scale_factor=ups, mode='bilinear')

    for idx, batch_x, batch_y in tqdm(dataloader, desc="Processing batches in dataloader"):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        # Generate batch mask
        if use_mask:
            if mask is None:
                raise ValueError("Mask must be provided when use_mask=True.")
            batch_mask = ups(mask[idx]).to(device).squeeze()
        else:
            if explanation_method is None:
                raise ValueError("Explanation method must be provided when use_mask=False.")
            batch_mask = explanation_method.saliency(batch_x, batch_y)

        # Generate ground truth mask
        groundtruth = torch.where(torch.sum(batch_x, 1) >= 2.99, 1, 0).to(device)
        p = torch.sum(groundtruth, dim=(1, 2))

        for i in tqdm(range(len(idx)), desc="Iterating through batch indices to calculate IOU", leave=False):
            top_p_ind = torch.sort(batch_mask[i].flatten())[0][-p[i]]
            im_mask = torch.where(batch_mask[i] >= top_p_ind, 1.0, 0.0)

            # IoU Calculation
            intersection = torch.sum(im_mask * groundtruth[i])
            union = torch.sum(torch.where(im_mask + groundtruth[i] >= 1.0, 1.0, 0.0))
            iou = intersection / union if union > 0 else 0.0
            ious.append(iou)

            # Collect a few samples for visualization (only if needed)
            if len(saved_samples) < num_samples:
                saved_samples.append((im_mask.cpu(), groundtruth[i].cpu(), idx[i].cpu()))

    # Save and visualize selected samples
    # for i, (im_mask, groundtruth, sample_idx) in enumerate(saved_samples):
    #     torch.save({
    #         "prediction": im_mask,
    #         "groundtruth": groundtruth,
    #         "index": sample_idx
    #     }, os.path.join(save_dir, f"sample_{sample_idx}.pt"))

        # visualize_side_by_side(im_mask, groundtruth, save_path=os.path.join(save_dir, f"sample_{sample_idx}.png"))

    ious = torch.tensor(ious)
    return torch.mean(ious), torch.std(ious)



def run_iou_for_methods(
    methods, dataloaders, model, device, save_dir, num_samples, diet_mask_paths=None, ups=None
):
    """
    Run IoU calculations for multiple explanation methods, including DIET.

    Args:
    - methods: Dictionary of explanation methods (e.g., {"GRAD": InputGradient, "DIET": None}).
    - dataloaders: Dictionary containing train and test dataloaders (e.g., {"train": train_loader, "test": test_loader}).
    - device: Torch device (CPU/GPU).
    - save_dir: Directory to save results.
    - num_samples: Number of samples to visualize.
    - diet_mask_paths: Dictionary containing paths to DIET masks (e.g., {"train": "path/to/train_mask.pt", "test": "path/to/test_mask.pt"}).
    - ups: Upsampling factor for DIET masks (if needed).

    Returns:
    - visualization_data: List of dictionaries with prediction and ground truth data for visualization.
    """
    os.makedirs(save_dir, exist_ok=True)
    visualization_data = []

    for method_name, method_class in methods.items():
        print(f"Running IoU for: {method_name}")

        for split_name, dataloader in dataloaders.items():
            # Handle DIET masks as a special case
            if method_name == "DIET":
                if diet_mask_paths is None or split_name not in diet_mask_paths:
                    raise ValueError(f"DIET masks not provided for {split_name}.")
                mask = torch.load(diet_mask_paths[split_name])
                prediction_func = lambda x, _: torch.nn.Upsample(scale_factor=ups, mode='bilinear')(mask).squeeze()
                mean_iou, std_iou = calculate_iou(
                    dataloader=dataloader,
                    device=device,
                    save_dir=os.path.join(save_dir, f"{method_name}_{split_name}"),
                    num_samples=num_samples,
                    use_mask=True,
                    mask=mask,
                    ups=ups
                )
            else:
                method = method_class(model)
                prediction_func = lambda x, y: method.saliency(x, y)
                mean_iou, std_iou = calculate_iou(
                    dataloader=dataloader,
                    device=device,
                    save_dir=os.path.join(save_dir, f"{method_name}_{split_name}"),
                    num_samples=num_samples,
                    use_mask=False,
                    explanation_method=method
                )

            print(f"{method_name} ({split_name}): Mean IoU = {mean_iou:.4f}, Std Dev = {std_iou:.4f}")

            # Collect predictions for visualization
            visualization_data.append({
                "method_name": f"{method_name} ({split_name})",
                "prediction_func": prediction_func,
                "split": split_name
            })

    return visualization_data



def visualize_methods(visualization_data, dataloaders, device, save_dir, num_samples):
    """
    Visualize predictions and ground truths for all methods and splits.

    Args:
    - visualization_data: List of dictionaries containing prediction functions and metadata.
    - dataloaders: Dictionary containing train and test dataloaders.
    - device: Torch device.
    - save_dir: Directory to save visualizations.
    """
    os.makedirs(save_dir, exist_ok=True)

    for split_name, dataloader in dataloaders.items():
        _, example_batch_x, example_batch_y = next(iter(dataloader))
        example_batch_x, example_batch_y = example_batch_x.to(device), example_batch_y.to(device)
        example_batch_x = example_batch_x[:num_samples]
        example_batch_y = example_batch_y[:num_samples]


        groundtruth = torch.where(torch.sum(example_batch_x, dim=1) >= 2.99, 1, 0)

        # Prepare predictions for the given split
        predictions = []
        method_names = []
        for data in visualization_data:
            if data["split"] == split_name:
                pred = data["prediction_func"](example_batch_x, example_batch_y)
                predictions.append(pred.cpu())
                method_names.append(data["method_name"])

        # Save visualization
        save_path = os.path.join(save_dir, f"{split_name}_comparison.png")
        visualize_explanations(predictions, groundtruth, method_names, save_path)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", default=2500, type=int, help="fraction of pixels to keep")
    parser.add_argument("--model_path", type=str, help="path to pretrained model")
    parser.add_argument("--mask_path", type=str, help="path to pretrained model")
    parser.add_argument("--mask_num", default='1', type=str, help="mask num path")
    parser.add_argument("-ups", default=16, type=int, help="upsample factor")

    args = parser.parse_args()

    device = "cuda"
    batch_sz = 128
    # p=args.p
    

    print(time.ctime().split(" ")[3], "loading data...", flush=True)

    data_dir = "data/hard_mnist/"
    train_imgs, train_labels, test_imgs, test_labels = load_mnist_from_disk(data_dir)
    num_classes = 10

    train_loader = torch.utils.data.DataLoader(DatasetfromDisk(train_imgs, train_labels), batch_size=batch_sz, shuffle=False)
    test_loader = torch.utils.data.DataLoader(DatasetfromDisk(test_imgs, test_labels), batch_size=batch_sz, shuffle=False)
    print(time.ctime().split(" ")[3], "finished loading data!", flush=True)

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights=None)
    model.fc = torch.nn.Linear(512, num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    model = model.to(device)
    model.eval()

    expl_methods = {
                    # "SMOOTHGRAD": SmoothGrad, 
                    # "GRADCAM":GradCAM, 
                    "GRAD":InputGradient, 
                    # "SimpleFullGrad":SimpleFullGrad
                    }


    # Explanation methods
    expl_methods = {
        "GRAD": InputGradient,
        "DIET": None  # Placeholder; DIET mask handling is separate
    }

    # Precomputed mask paths for DIET
    mask_paths = {
        "train":args.mask_path + "/mask_" + args.mask_num + ".pt",
        "test": args.mask_path + "/test_mask.pt"
    }

    # Dataloaders
    dataloaders = {
        "train": train_loader,
        "test": test_loader
    }

    # Run IoU calculations and collect data for visualization
    visualization_data = run_iou_for_methods(
        methods=expl_methods,
        dataloaders=dataloaders,
        model=model,
        device=device,
        save_dir="results",
        num_samples=5,
        diet_mask_paths=mask_paths,
        ups=4
    )

    # Visualize results
    visualize_methods(
        visualization_data=visualization_data,
        dataloaders=dataloaders,
        device=device,
        save_dir="results",
        num_samples=5
    )



    # for method_name in expl_methods:
    #     print(method_name)
    #     expl_method = expl_methods[method_name](model)

    #     print(time.ctime().split(" ")[3], "(" + method_name + ") simplifying data...")

    #     train_iou = mnist_iou(expl_method, train_loader, device)
    #     test_iou = mnist_iou(expl_method, test_loader, device)

    #     print(time.ctime().split(" ")[3], "(" + method_name + ") finished simplifying data!", flush=True)
    #     print(time.ctime().split(" ")[3], "(" + method_name + ") train iou", train_iou, "test iou", test_iou, flush=True)

    # train_mask = torch.load(args.mask_path + "/mask_" + args.mask_num + ".pt")
    # test_mask = torch.load(args.mask_path + "/test_mask.pt")
    # train_iou = our_iou(train_mask, train_loader, args.ups, device)
    # test_iou = our_iou(test_mask, test_loader, args.ups, device)
    # print(time.ctime().split(" ")[3], "(OURS) train", train_iou, "test", test_iou, flush=True)





if __name__ == "__main__":
    main()






        




