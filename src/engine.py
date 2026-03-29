import os
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score
from timm.data.mixup import Mixup

@torch.no_grad()
def evaluate(model, dataloader, loss_fn, device, desc="Val"):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_targets = []

    pbar = tqdm(dataloader, desc=desc, leave=False)
    for X_batch, y_batch in pbar:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = loss_fn(logits, y_batch)

        batch_size = y_batch.size(0)
        total_loss += loss.item() * batch_size

        y_pred = logits.argmax(dim=1)
        total_correct += (y_pred == y_batch).sum().item()
        total_samples += batch_size

        avg_loss = total_loss / max(total_samples, 1)
        acc = total_correct / max(total_samples, 1)

        pbar.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.4f}")

        all_preds.extend(y_pred.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)

    f1 = f1_score(all_targets, all_preds, average='macro')
    return accuracy, avg_loss, f1

def train_model(model, loss_fn, optimizer, train_loader, val_loader, device,\
         n_epoch=3, lr_eta=1e-6, ls=0.1):
    best_val_f1 = 0.0
    best_val_loss = 100
    num_iter = 0
    counter_early_stop = 0
    save_path = f"checkpoints/best_model.pth"
    scheduler = CosineAnnealingLR(optimizer, T_max=n_epoch, eta_min=lr_eta) # планировщик скорости обучения
    loss_fn_val = torch.nn.CrossEntropyLoss(label_smoothing=ls)
    mixup_fn = Mixup(
        mixup_alpha=0.4,
        cutmix_alpha=1.0,
        prob=1.0,
        switch_prob=0.5,
        label_smoothing=0.1,
        num_classes=15
    )

    for epoch in range(1, n_epoch + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        pbar = tqdm(train_loader, desc=f"Ep {epoch}/{n_epoch}", leave=True, dynamic_ncols=True)

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            X_batch, y_batch = mixup_fn(X_batch, y_batch)

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size = y_batch.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            y_true_cls = y_batch.argmax(dim=1)
            y_pred_cls = logits.argmax(dim=1)

            correct = (y_pred_cls == y_true_cls).sum().item()
            total_correct += correct

            avg_loss = total_loss / max(total_samples, 1)
            acc = total_correct / max(total_samples, 1)

            current_lr = optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f"{avg_loss:.4f}",
                'acc': f"{acc:.4f}",
                'lr': f"{current_lr:.6f}"
            })

            num_iter += 1

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        val_acc, val_loss, val_f1 = evaluate(model, val_loader, loss_fn_val, device, desc=f"Val {epoch}/{n_epoch}")

        print(f"Epoch {epoch}/{n_epoch}: val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        # Проверка на лучшую модель
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            counter_early_stop = 0

            torch.save(model.state_dict(), save_path)
            print(f"--- Эпоха {epoch}: Новая лучшая точность f1: {val_f1:.4f}! Модель сохранена. ---")
        elif (val_f1 == best_val_f1) and (val_loss < best_val_loss):
            best_val_loss = val_loss
            counter_early_stop = 0

            torch.save(model.state_dict(), save_path)
            print(f"--- Эпоха {epoch}: Новая лучшая точность f1: {val_f1:.4f}! Модель сохранена. ---")
        else: # Early stop
            counter_early_stop += 1
            if counter_early_stop >= 5:
                print(f"Сработал Earle stop!")
                break

    # Загружаем лучший вес
    model.load_state_dict(torch.load(f"checkpoints/best_model.pth"))
    os.remove(f"checkpoints/best_model.pth")
    return model