import os
import glob
import torch
import timm
import numpy as np
import cv2
import argparse
from src.config import DEVICE, MODEL_NAME, NUM_CLASSES, IDX_TO_CLASS, FOLD_SCORES
from src.dataset import get_val_transforms

@torch.no_grad()
def predict_ensemble(image_path, checkpoints_dir):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    model.eval()

    weight_models = sorted(glob.glob(os.path.join(checkpoints_dir, "*.pth")))
    print(f"Найдено моделей: {len(weight_models)}")

    # Подготовка изображения
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = get_val_transforms()
    img_tensor = transform(image=image)['image'].unsqueeze(0).to(DEVICE)

    # TTA тензоры
    img_h = torch.flip(img_tensor, [3])
    img_v = torch.flip(img_tensor, [2])
    img_hv = torch.flip(img_tensor, [2, 3])
    tta_batch = torch.cat([img_tensor, img_h, img_v, img_hv], dim=0)

    all_probs = []

    for w in weight_models:
        model.load_state_dict(torch.load(w, map_location=DEVICE))
        with torch.no_grad():
            logits = model(tta_batch)
            probs = torch.softmax(logits, dim=1).mean(dim=0)
            all_probs.append(probs.cpu().numpy())

    weighted_probs = np.zeros(NUM_CLASSES)

    for i, prob in enumerate(all_probs):
        weighted_probs += prob * FOLD_SCORES[i]

    weighted_probs /= sum(FOLD_SCORES)
    pred_idx = weighted_probs.argmax()
    confidence = weighted_probs[pred_idx]
    
    return IDX_TO_CLASS[pred_idx], confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Путь к картинке")
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints/", help="Папка с весами моделей")
    args = parser.parse_args()

    class_name, conf = predict_ensemble(args.image, args.ckpt_dir)
    
    print(f"\nРезультат: {class_name}")
    print(f"Уверенность: {conf * 100:.2f}%")