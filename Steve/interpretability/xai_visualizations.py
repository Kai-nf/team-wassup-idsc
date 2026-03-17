import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
import torch.nn.functional as F


import sys
from pathlib import Path

# 1. Define the root of your project
REPO_ROOT = Path(__file__).resolve().parents[2]

# 2. Add the project root to Python's system path so it can "see" the Steve folder
sys.path.append(str(REPO_ROOT))

# 3. NOW it is safe to import from Steve!
from Steve.models.models_1d_cnn import ECG1DCNN


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks to grab the mathematical arrays inside the CNN
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
        
    def save_activation(self, module, input, output):
        self.activations = output.detach()
        
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
        
    def generate_cam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        
        # Grab the prediction score
        score = output.squeeze() 
        
        self.model.zero_grad()
        score.backward(retain_graph=True)
        
        # Global Average Pooling on the gradients across the temporal dimension
        weights = torch.mean(self.gradients, dim=2, keepdim=True)
        
        # Weighted combination of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)
        
        # Interpolate the heatmap back to the original 4000-step sequence length
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)
        
        # Normalize between 0 and 1 so it plots nicely as a color gradient
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy()

def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    vis_dir = REPO_ROOT / "results" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths
    data_path = REPO_ROOT / "JiaKang" / "dataset_v3_wavelet.npy"
    manifest_path = REPO_ROOT / "JiaKang" / "master_folds_drop14.json"
    weights_path = REPO_ROOT / "results" / "model_weights" / "1dcnn_fold_0.pt"
    preds_path = REPO_ROOT / "results" / "method3_1dcnn_predictions.json"
    
    # Load Model
    model = ECG1DCNN()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    else:
        print(f"Error: Could not find model weights at {weights_path}")
        return
        
    model.eval()
    
    # Automatically find the last Conv1d layer
    target_layer = None
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv1d):
            target_layer = module
            break
            
    if target_layer is None:
        raise ValueError("Could not find a Conv1d layer in the model.")
        
    grad_cam = GradCAM1D(model, target_layer)
    
    # Identify True Positives (Brugada label=1, prob>0.8) from Fold 0
    with open(preds_path, 'r') as f:
        preds = json.load(f)
        
    fold_0_data = preds["fold_0"]
    tp_patients = []
    for pid, y_true, y_prob in zip(fold_0_data["patient_ids"], 
                                   fold_0_data["y_true_patient"], 
                                   fold_0_data["y_prob_patient"]):
        if y_true == 1 and y_prob > 0.8:
            tp_patients.append((pid, y_prob))
            
    # Select up to 5 patients to visualize
    target_patients = tp_patients[:5]
    if not target_patients:
        print("No True Positives found matching the criteria.")
        return
        
    # Load the massive wavelet array
    wavelets = np.load(data_path)
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # --- TEMPORARY MAPPING LOGIC ---
    # We grab all patients in Fold 0 to map them to the .npy rows. 
    # (If this throws an IndexError later, we will update this to use v3_fold_composition.json)
    all_pids_fold0 = manifest["folds"]["0"]["test"]["patient_ids"] + manifest["folds"]["0"]["train"]["patient_ids"]
    pid_to_idx = {int(pid): idx for idx, pid in enumerate(all_pids_fold0)}
    
    for pid, prob in target_patients:
        try:
            idx = pid_to_idx[int(pid)]
            
            # 1. Grab the signal
            signal = wavelets[idx] 
            
            # 2. ADD THIS LINE: Transpose from [Length, Channels] to [Channels, Length]
            signal = np.transpose(signal)
            
        except (KeyError, IndexError):
            print(f"Skipping Patient {pid}: Index mapping needs v3_fold_composition.json.")
            continue
            
        # Now this will perfectly become [1, 12, 101] for PyTorch!
        tensor_signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        
        # Generate the Heatmap
        cam = grad_cam.generate_cam(tensor_signal)
        
        # Plotting
        seq_len = signal.shape[-1]
        x_axis = np.arange(seq_len)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Plot the first channel of the wavelet/ECG signal
        ax.plot(x_axis, signal[0], color='blue', linewidth=1.5, alpha=0.8, label='Signal (Ch 0)')
        
        # Overlay the heatmap properly using imshow
        cam_2d = cam.reshape(1, -1)
        ax.imshow(cam_2d, cmap='Reds', aspect='auto', alpha=0.5, 
                  extent=[0, seq_len, signal[0].min(), signal[0].max()])
        
        ax.set_title(f"Grad-CAM 1D | Patient ID: {pid} | Pred Prob: {prob:.3f} (Brugada)")
        ax.set_xlabel("Time / Sequence Steps")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right')
        
        out_file = vis_dir / f"gradcam_patient_{pid}.png"
        plt.tight_layout()
        plt.savefig(out_file, dpi=300)
        plt.close()
        
        print(f"Saved Grad-CAM for Patient {pid} to {out_file}")

if __name__ == "__main__":
    main()