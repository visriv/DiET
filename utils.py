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