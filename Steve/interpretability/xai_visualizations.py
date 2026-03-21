import os
import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Define the root of your project
REPO_ROOT = Path(__file__).resolve().parents[2]

# 2. Add project root to Python path (still useful for other imports)
sys.path.append(str(REPO_ROOT))


def _load_ecg1dcnn_class(repo_root: Path):
    """
    Robust loader for ECG1DCNN after file move/rename.
    Works even when filenames start with numbers (e.g., 1d_cnn_beatlevel.py).
    Priority:
      1) model/1d_cnn_beat_level.py
      2) model/1d_cnn_patient_level.py
    """
    candidate_files = [
        repo_root / "model" / "1d_cnn_beat_level.py",
        repo_root / "model" / "1d_cnn_patient_level.py",
    ]

    errors = []
    for py_file in candidate_files:
        if not py_file.exists():
            errors.append(f"Not found: {py_file}")
            continue

        try:
            spec = importlib.util.spec_from_file_location("ecg1dcnn_module", str(py_file))
            if spec is None or spec.loader is None:
                errors.append(f"Invalid import spec: {py_file}")
                continue

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "ECG1DCNN"):
                print(f"Loaded ECG1DCNN from: {py_file}")
                return module.ECG1DCNN
            else:
                errors.append(f"No ECG1DCNN class in: {py_file}")
        except Exception as e:
            errors.append(f"Import failed for {py_file}: {repr(e)}")

    raise ImportError(
        "Could not load ECG1DCNN.\n"
        + "\n".join(errors)
    )


# Load class dynamically
ECG1DCNN = _load_ecg1dcnn_class(REPO_ROOT)


class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks to grab arrays inside the CNN
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)

        score = output.squeeze()
        self.model.zero_grad()
        score.backward(retain_graph=True)

        # Global average pooling over temporal dimension
        weights = torch.mean(self.gradients, dim=2, keepdim=True)

        # Weighted sum of feature maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize to original sequence length
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode="linear", align_corners=False)

        # Normalize [0,1]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


# ... keep the rest of your file unchanged ...


def _find_last_conv1d_layer(model):
    for module in reversed(list(model.modules())):
        if isinstance(module, torch.nn.Conv1d):
            return module
    return None


def _pick_case(patient_ids, y_true, y_prob, mode):
    """
    mode:
      - 'TP': y_true==1 and y_prob>0.8
      - 'FP': y_true==0 and y_prob>0.8
      - 'FN': y_true==1 and y_prob<0.2
    Returns tuple(pid, y_true, y_prob) or None
    """
    for pid, yt, yp in zip(patient_ids, y_true, y_prob):
        yt_i = int(yt)
        yp_f = float(yp)

        if mode == "TP" and yt_i == 1 and yp_f > 0.8:
            return (pid, yt_i, yp_f)
        if mode == "FP" and yt_i == 0 and yp_f > 0.8:
            return (pid, yt_i, yp_f)
        # Change this line:
        if mode == "FN" and yt_i == 1 and yp_f < 0.5:  # Relaxed to 0.5
            return (pid, yt_i, yp_f)
    return None


def _save_gradcam_plot(signal, cam, pid, y_true, y_prob, label_long, out_file):
    seq_len = signal.shape[-1]
    x_axis = np.arange(seq_len)

    fig, ax = plt.subplots(figsize=(10, 4))

    # Plot first channel
    ax.plot(x_axis, signal[0], color="blue", linewidth=1.5, alpha=0.85, label="Signal (Ch 0)")

    # Overlay CAM
    cam_2d = cam.reshape(1, -1)
    ax.imshow(
        cam_2d,
        cmap="Reds",
        aspect="auto",
        alpha=0.5,
        extent=[0, seq_len, signal[0].min(), signal[0].max()],
    )

    ax.set_title(
        f"Grad-CAM | {label_long} | Patient ID: {pid} | y_true={y_true} | y_prob={y_prob:.3f}"
    )
    ax.set_xlabel("Time / Sequence Steps")
    ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(out_file, dpi=300)
    plt.close()


def main():
    REPO_ROOT = Path(__file__).resolve().parents[2]
    vis_dir = REPO_ROOT / "results" / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    # 1. FIXED PATHS: Point to Preprocessed_Dataset and v3.1
    data_path = REPO_ROOT / "Preprocessed_Dataset" / "dataset_v3.1_wavelet.npy"
    comp_path = REPO_ROOT / "Preprocessed_Dataset" / "fold_composition_v3.1.json"
    weights_path = REPO_ROOT / "results" / "model_weights" / "1dcnn_fold_0.pt"

    # Use your requested ensemble JSON
    preds_path_primary = REPO_ROOT / "results" / "ensemble_CNN065_LR035.json"

    # Load model
    model = ECG1DCNN()
    if weights_path.exists():
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
    else:
        print(f"Error: Could not find model weights at {weights_path}")
        return
    model.eval()

    target_layer = _find_last_conv1d_layer(model)
    if target_layer is None:
        raise ValueError("Could not find a Conv1d layer in the model.")

    grad_cam = GradCAM1D(model, target_layer)

   # ---------- Load the .npy prediction files we saved during training ----------
    fold_dir = REPO_ROOT / "results" / "fold_outputs"
    pid_file = fold_dir / "fold_0_patient_ids.npy"
    yt_file  = fold_dir / "fold_0_y_true.npy"
    yp_file  = fold_dir / "fold_0_y_prob.npy"

    if not (pid_file.exists() and yt_file.exists() and yp_file.exists()):
        raise ValueError(f"Error: Could not find .npy files in {fold_dir}. Make sure the training script finished successfully.")

    patient_ids = np.load(pid_file)
    y_true = np.load(yt_file)
    y_prob = np.load(yp_file)

    if not (len(patient_ids) == len(y_true) == len(y_prob)):
        raise ValueError("Length mismatch in the loaded .npy arrays!")
           

    # ---------- Find requested cases ----------
    tp_case = _pick_case(patient_ids, y_true, y_prob, mode="TP")
    fp_case = _pick_case(patient_ids, y_true, y_prob, mode="FP")
    fn_case = _pick_case(patient_ids, y_true, y_prob, mode="FN")

    # Informative logs (important for judges/audit trail)
    print("Case selection status:")
    print(" TP found:", tp_case is not None)
    print(" FP found:", fp_case is not None, "(may be none because fold_0 FP=0 at t=0.5)")
    print(" FN found:", fn_case is not None)

    # Load wavelets + beat mapping JSON
    if not data_path.exists():
        print(f"Error: Could not find data at {data_path}")
        return
    if not comp_path.exists():
        print(f"Error: Could not find composition manifest at {comp_path}")
        return

    wavelets = np.load(data_path)
    with open(comp_path, "r", encoding="utf-8") as f:
        comp_data = json.load(f)
        
    # 2. FIXED MAPPING: Load the actual beat row assignments
    beat_pids = np.array(comp_data["patient_ids"])

    targets = [
        ("TP", "True Positive (Success)", tp_case, "gradcam_TP_patient_{pid}.png"),
        ("FP", "False Positive (False Alarm)", fp_case, "gradcam_FP_patient_{pid}.png"),
        ("FN", "False Negative (Missed Case)", fn_case, "gradcam_FN_patient_{pid}.png"),
    ]

    generated = 0
    for short_label, long_label, case, out_pattern in targets:
        if case is None:
            print(f"Warning: No {long_label} found in fold 0 with required criteria.")
            continue

        pid, yt, prob = case

        try:
            # Safely find all beats belonging to this patient and grab the first one
            beat_indices = np.where(beat_pids == int(pid))[0]
            if len(beat_indices) == 0:
                print(f"Skipping {long_label} for Patient {pid}: No beats found in array.")
                continue
                
            idx = beat_indices[0] 
            signal = wavelets[idx]
            signal = np.transpose(signal)  # [Length, Channels] -> [Channels, Length]
        except (KeyError, IndexError, ValueError):
            print(f"Skipping {long_label} for Patient {pid}: mapping/index issue.")
            continue

        tensor_signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0)
        cam = grad_cam.generate_cam(tensor_signal)

        out_file = vis_dir / out_pattern.format(pid=pid)
        _save_gradcam_plot(
            signal=signal,
            cam=cam,
            pid=pid,
            y_true=yt,
            y_prob=prob,
            label_long=long_label,
            out_file=out_file,
        )

        print(f"Saved {long_label} Grad-CAM for Patient {pid} -> {out_file}")
        generated += 1

    if generated == 0:
        print("No Grad-CAM plots were generated. Check criteria or JSON array keys.")
    else:
        print(f"\n✅ Done. Generated {generated} Grad-CAM figure(s) in: {vis_dir}")

if __name__ == "__main__":
        main()