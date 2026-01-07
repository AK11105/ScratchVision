if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import torch.optim as optim
    
    from src.models import VGGNet
    from src.trainers.vggnet.train_vggnet import train
    from data.download_datasets import download_Imagenette
    from src.utils.loaders import create_DataLoaders
    from src.utils.visualization import *
    
    BATCH_SIZE=64
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    train_dataset, test_dataset = download_Imagenette()
    train_dataloader, test_dataloader = create_DataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle_train=True,
        num_workers=0
    )
    
    model = VGGNet(output_clases = 10)
    
    criterion = nn.CrossEntropyLoss()
    
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith("bias"):
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = torch.optim.SGD(
        [
            {"params": decay, "weight_decay": 5e-4},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=0.01,
        momentum=0.9,
    )
    
    train_losses, train_top1_errors, train_top3_errors, train_top5_errors, \
    test_losses, test_top1_errors, test_top3_errors, test_top5_errors = train(
        num_epochs=35,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader, 
        criterion=criterion,
        optimizer=optimizer,  # Note: optimizer= (match train() param name)
        device=device,
        update_freq=10
    )