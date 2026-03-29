import torch
import timm
from timm.loss import SoftTargetCrossEntropy
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold

from src.config import *
from src.dataset import collect_subclass_data, MyDataset, get_train_transforms, get_val_transforms
from src.utils import seed_everything, sklearn_report
from src.engine import train_model

def main():
    seed_everything(SEED)
    dataset_path = TRAIN_PATH 
    
    print("Сбор данных")
    all_files, all_labels = collect_subclass_data(dataset_path)
    
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    loss_fn = SoftTargetCrossEntropy()

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_files, all_labels)):
            
        print(f"\n{'='*30}\n ЗАПУСК ФОЛДА {fold_idx + 1}/{N_FOLDS}\n{'='*30}")

        train_dataset = MyDataset(all_files[train_idx], CLASS_TO_IDX, get_train_transforms())
        val_dataset = MyDataset(all_files[val_idx], CLASS_TO_IDX, get_val_transforms())

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True)

        model = timm.create_model(
            MODEL_NAME,
            pretrained=True,
            num_classes=NUM_CLASSES,
            drop_rate=0.2,
            drop_path_rate=0.1
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)

        model = train_model(model, loss_fn, optimizer, train_loader, val_loader, DEVICE, EPOCHS, LR_ETA, LS)

        print(f"\nОтчет для фолда {fold_idx + 1}:")
        sklearn_report(model, val_loader, DEVICE, IDX_TO_CLASS)

        save_path = f"checkpoints/{MODEL_NAME}_fold_{fold_idx+1}.pth"
        torch.save(model.state_dict(), save_path)
        print(f"Модель фолда сохранена: {save_path}")

        del model, optimizer, train_loader, val_loader
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()