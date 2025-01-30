import matplotlib.pyplot as plt
def visualize_side_by_side(prediction, groundtruth, save_path=None):
    """
    Visualize prediction and ground truth side by side.

    Args:
    - prediction: Predicted mask (tensor or numpy array).
    - groundtruth: Ground truth mask (tensor or numpy array).
    - save_path: Path to save the visualization image (optional).
    """
    prediction = prediction.numpy().squeeze()
    groundtruth = groundtruth.numpy().squeeze()

    # Create a figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    
    axs[0].imshow(prediction, cmap='viridis')
    axs[0].set_title("Prediction")
    axs[0].axis("off")

    axs[1].imshow(groundtruth, cmap='viridis')
    axs[1].set_title("Ground Truth")
    axs[1].axis("off")
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def visualize_explanations(predictions, groundtruth, method_names, save_path=None):
    """
    Visualize predictions and ground truth for multiple methods.

    Args:
    - predictions: List of predicted masks from different methods.
    - groundtruth: Ground truth mask.
    - method_names: List of method names corresponding to predictions.
    - save_path: Path to save the visualization image (optional).
    """
    if not isinstance(predictions, list):
        predictions = [predictions]
    if not isinstance(method_names, list):
        method_names = [method_names]

    num_rows = len(method_names)
    num_samples = groundtruth.shape[0]

    fig, axs = plt.subplots(num_rows*num_samples, 2, figsize=(10, 5 * num_rows))

    for method_idx, method_name in enumerate(method_names):
        for sample_idx in range(num_samples):
            row = method_idx * num_samples + sample_idx

            # Prediction
            axs[row, 0].imshow(predictions[method_idx][sample_idx].cpu().detach().numpy().squeeze(), cmap='viridis')
            axs[row, 0].set_title(f"{method_name} - Prediction (Sample {sample_idx + 1})")
            axs[row, 0].axis("off")

            # Ground Truth
            axs[row, 1].imshow(groundtruth[sample_idx].cpu().detach().numpy().squeeze(), cmap='viridis')
            axs[row, 1].set_title(f"{method_name} - Ground Truth (Sample {sample_idx + 1})")
            axs[row, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    