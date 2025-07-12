""" Here is the data I collected over 11 trials. It is more easily viewed in the Vision_Transformer_from_scratch.ipynb file under 'notebook'. """

data = {
    'Trial #': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    'Image size': [32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64],
    'Embedding dimension': [384, 384, 384, 192, 192, 192, 192, 252, 252, 252, 252],
    'Number of heads': [12, 12, 12, 12, 12, 8, 8, 12, 12, 12, 12],
    'Number of encoders': [8, 8, 8,  8, 8, 8, 6, 8, 8, 8, 12],
    'Batch size': [128, 128, 128, 128, 128, 128, 128, 128, 128, 512, 512],
    'Epochs': [20, 20, 50,  30, 30, 30, 30, 100, 100, 100, 100],
    'Patch size': [4, 4, 4,  4, 8, 4, 4, 4, 4, 4, 4],
    'Optimizer': ['SGD', 'Adam', 'SGD',  "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW", "AdamW"],
    'Learning rate': [0.001, 0.001, 0.1,  3e-4, 3e-4, 3e-4, "3e-4 with 1e-5 weight decay", "0.003 with 0.05 weight decay", "0.003 with 0.05 weight decay", "0.003 with 0.05 weight decay", "0.003 with 0.05 weight decay"],
    'Normalization': ['No', 'No', 'Yes',  'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes'],
    'Loss function': ['Cross entropy', 'Cross entropy', 'Cross entropy',  'Cross entropy', 'Cross entropy', 'Cross entropy', 'Cross entropy', 'Cross entropy + soft cross entropy', 'Cross entropy + soft cross entropy', 'Cross entropy + soft cross entropy', 'Cross entropy + soft cross entropy'],
    'Dropout regularization value': [None, 0.15, 0.15, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05],
    'Data Augmentation': [None, None, "Jittering + horizontal flip", "Jittering + random crop + horizontal flip", "Jittering + random crop + horizontal flip",  "Jittering + random crop + horizontal flip", "Jittering + random crop + horizontal flip", "Jittering + random crop + horizontal flip + mild mixup starting at epoch 20", "Jittering + random crop + horizontal flip + mild mixup starting at epoch 20" , "Jittering + random crop + horizontal flip + mild mixup starting at epoch 10", "Jittering + random crop + horizontal flip + mild mixup starting at epoch 10 + mild cutmix starting at epoch 20"],
    'Scheduler': [None, None, None, "Cosine Annealing LR", "Cosine Annealing LR", "Cosine Annealing LR", "Cosine Annealing LR", "Cosine Annealing LR + linear warmup", "Cosine Annealing LR + linear warmup", "Cosine Annealing LR + linear warmup", "Custom scheduler implementation: Linear warmup + hold for 30 epochs + cosine decay"],
    'Accuracy %': [23.24, 23.24, 33.36, 10.00, 37.53,  70.49, 71.44, 75.69, 83.07, 84.21, 85.68],
}
