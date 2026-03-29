import torch
from pathlib import Path

TRAIN_PATH = Path(r"data\train\train") 

SEED = 9999
IMG_SIZE = 300
NUM_CLASSES = 15

# Настройки обучения
BATCH_SIZE = 16
EPOCHS = 25
LR = 3e-4
LR_ETA = 1e-6
LS = 0.1
N_FOLDS = 6
MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_TO_IDX = {
    "Апельсин": 0, "Бананы": 1, "Груши": 2, "Кабачки": 3,
    "Капуста": 4, "Картофель": 5, "Киви": 6, "Лимон": 7,
    "Лук": 8, "Мандарины": 9, "Морковь": 10, "Огурцы": 11,
    "Томаты": 12, "Яблоки зеленые": 13, "Яблоки красные": 14
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

FOLD_SCORES = [0.9756, 0.9731, 0.9655, 0.9708, 0.9718, 0.9759]
sum_scores = sum(FOLD_SCORES)
FOLD_WEIGHTS = [score / sum_scores for score in FOLD_SCORES]