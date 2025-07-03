import torch
import torch.nn as nn
import numpy as np
from net import *
from dataset import *
import torchvision.transforms as transforms
from PIL import Image
import torch.backends.cudnn as cudnn
import time
from net import *

# system-related parameters
data_dir = "./data"
train_dir = data_dir + "/train.txt"
val_dir = data_dir + "/test.txt"
image_dir = data_dir + "/Images/"
binary_map_dir = data_dir + "/masks/"
phragmite_color = [61, 245, 61]
image_scale = 1. / 255
image_mean = [0.4663, 0.4657, 0.3188]
image_std = [1, 1, 1]
image_mean = np.array(image_mean).reshape((1, 1, 3))
image_std = np.array(image_std).reshape((1, 1, 3))
ratio = 0.125

# === Simplified Checkpoint Loading ===
def load_model(checkpoint_path, model, device="cuda"):
    """Loads model weights from a checkpoint for inference."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'],strict = False)
    model.to(device).eval().half()  # Move to GPU and set to evaluation mode
    print(f"✅ Model loaded from '{checkpoint_path}'")
    return model

def run_dummy_input(model,dataloader,device = "cuda"):
    model.eval()
    sample = next(iter(dataloader))
    image = sample['image'].to(device)

    dummy_input = torch.randn_like(image).to(device).half()

    for _ in range(10):
        with torch.no_grad():
            _ = model(dummy_input)
    
    print("Dummy input inference complete. Model warmed up!")

# === Inference Function ===
def run_inference(model, dataloader, device="cuda"):
    """Runs inference on the dataset and returns predicted maps."""
    model.eval()
    predicted_map = {}

    num_frames = len(dataloader)
    total_time = 0.0
    with torch.no_grad():  # Disable gradients for inference
        for sample in dataloader:
            start = time.time()
            image, file_name = sample['image'], sample['file_name'][0]
            print(f"Processing {file_name}...")

            # Run inference
            image = image.cuda().half()
            # logits = model(image).logits
            output = model(image)
            # output = torch.sigmoid(output)
            # output = F.interpolate(probs, size=image.shape[-2:], mode='bilinear', align_corners=True)
            output = output.squeeze().cpu().numpy()

            # Binarize output
            output[output >= 0.5] = 1
            output[output < 0.5] = 0

            # Store result
            predicted_map[file_name] = output
            print(f"✅ Inference completed for {file_name}")
            end = time.time()
            total_time += (end - start)

    fps = num_frames/total_time
    print("✅ All images processed successfully!")
    print(f"Inference Speed: {fps:.2f} FPS")
    return predicted_map

# === Main Execution ===
if __name__ == "__main__":
    cudnn.benchmark = True  # Optimizes CUDA performance
    torch.cuda.empty_cache()  

    # Load dataset and DataLoader
    val_transforms = transforms.Compose([
        Normalize(scale=image_scale, std=image_std, mean=image_mean, train=False),
        ToTensor(train=False),
        ZeroPadding(8, train=False)
    ])
    valset = PhragmiteDataset(
        data_dir=data_dir,
        data_list=val_dir,
        image_dir=image_dir,
        binary_map_dir=binary_map_dir,
        phragmite_color=phragmite_color,
        ratio=ratio,
        train=False,
        transform=val_transforms
    )
    val_loader = DataLoader(
        valset,batch_size=1,shuffle=False,num_workers=4,pin_memory=True
    )

    # Load model
    # model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512",
    #                                                          num_labels=1,ignore_mismatched_sizes=True)
    model = MobileNetV4Segmentation()

    # Load trained weights
    checkpoint_path = "./results/models/mobilenetv4_small/model_best_r1-score.pth"
    model = load_model(checkpoint_path, model)

    # GPU warm up
    run_dummy_input(model,val_loader)

    # Run inference
    num_frames = len(val_loader)
    predicted_maps = run_inference(model, val_loader)




