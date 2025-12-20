import os
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from models import get_model

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Hooks capture forward activations and backward gradients at `target_layer`.
        # These are later combined to produce the class-specific heatmap.
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):

        self.activations = output.detach().clone()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_idx):
        self.model.eval()
        self.model.zero_grad()
        
        with torch.set_grad_enabled(True):
            output = self.model(input_tensor)
            loss = output[0, class_idx]
            loss.backward()

        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        heatmap = torch.sum(weights * self.activations, dim=1, keepdim=True)

        heatmap = F.relu(heatmap)
        if torch.max(heatmap) > 0:
            heatmap /= torch.max(heatmap)
            
        return heatmap.detach().cpu().numpy().squeeze()

def visualize_result(img_path, heatmap, save_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    display = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    cv2.imwrite(save_path, display)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_name = 'resnet50' 
    
    print(f"Loading {model_name} for analysis...")
    model = get_model(model_name, num_classes=5)
    
    weights_path = f'outputs/{model_name}/best_model.pth'
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print("Model weights loaded.")
    
    model.to(device)

    # For ResNet, use the last block of `layer4` to get high-level conv features.
    target_layer = model.layer4[-1]

    cam = GradCAM(model, target_layer)

    sample_path = "data/CheXpert-v1.0-small/valid/patient64541/study1/view1_frontal.jpg"
    
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_img = Image.open(sample_path).convert('RGB')
    input_tensor = preprocess(input_img).unsqueeze(0).to(device)

    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural_Effusion']
    
    # Output: writes simple overlaid heatmaps to `outputs/<model>/gradcam_<label>.jpg`.
    print(f"Saving ResNet Heatmaps to outputs/{model_name}/...")
    for idx, name in enumerate(pathologies):
        heatmap = cam.generate_heatmap(input_tensor, class_idx=idx)
        save_name = f"outputs/{model_name}/gradcam_{name}.jpg"
        visualize_result(sample_path, heatmap, save_name)
        print(f" - Saved: {name}")