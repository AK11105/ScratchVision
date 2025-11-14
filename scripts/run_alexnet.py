if __name__ == "__main__":
    import torch
    import os 
    import yaml 
    import torch.nn as nn
    import torch.optim as optim
    
    from src.models.AlexNet import AlexNet
    from data.download_datasets import download_Imagenette
    from src.utils.loaders import create_DataLoaders
    from src.utils.visualization import *
    from src.trainers.alexnet import train
    
    with open('configs/alexnet.yaml', "r") as alexnet:
        config = yaml.safe_load(alexnet)
        
    batch_size = config["training"]["batch_size"]
    shuffle_train = config["training"]["shuffle_train"]
    num_workers = config["training"]["num_workers"]
    
    train_dataset, test_dataset = download_Imagenette()
    train_dataloader, test_dataloader = create_DataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers
    )
    
    model_cfg = config["model"]
    model = AlexNet(
        activation=None,
        pool=None,
        norm=None,
        dropout=None,

        C1_in       = model_cfg["conv1"]["in_channels"],
        C1_out      = model_cfg["conv1"]["out_channels"],
        C1_kernel   = model_cfg["conv1"]["kernel_size"],
        C1_stride   = model_cfg["conv1"]["stride"],
        C1_padding  = model_cfg["conv1"]["padding"],

        C2_in       = model_cfg["conv2"]["in_channels"],
        C2_out      = model_cfg["conv2"]["out_channels"],
        C2_kernel   = model_cfg["conv2"]["kernel_size"],
        C2_stride   = model_cfg["conv2"]["stride"],
        C2_padding  = model_cfg["conv2"]["padding"],

        C3_in       = model_cfg["conv3"]["in_channels"],
        C3_out      = model_cfg["conv3"]["out_channels"],
        C3_kernel   = model_cfg["conv3"]["kernel_size"],
        C3_stride   = model_cfg["conv3"]["stride"],
        C3_padding  = model_cfg["conv3"]["padding"],

        C4_in       = model_cfg["conv4"]["in_channels"],
        C4_out      = model_cfg["conv4"]["out_channels"],
        C4_kernel   = model_cfg["conv4"]["kernel_size"],
        C4_stride   = model_cfg["conv4"]["stride"],
        C4_padding  = model_cfg["conv4"]["padding"],

        C5_in       = model_cfg["conv5"]["in_channels"],
        C5_out      = model_cfg["conv5"]["out_channels"],
        C5_kernel   = model_cfg["conv5"]["kernel_size"],
        C5_stride   = model_cfg["conv5"]["stride"],
        C5_padding  = model_cfg["conv5"]["padding"],

        FC1_in      = model_cfg["fc1"]["in_features"],
        FC1_out     = model_cfg["fc1"]["out_features"],

        FC2_in      = model_cfg["fc2"]["in_features"],
        FC2_out     = model_cfg["fc2"]["out_features"],

        FC3_in      = model_cfg["fc3"]["in_features"],
        FC3_out     = model_cfg["fc3"]["out_features"],
    )
    
    criterion = nn.CrossEntropyLoss()
    
    optimizer_cfg = config["training"]["optimizer"]
    
    optimizer = optim.SGD(
        model.parameters(),
        momentum=optimizer_cfg["momentum"],
        weight_decay=optimizer_cfg["weight_decay"],
        lr=optimizer_cfg["lr"]
    )
    
    train_losses, train_top1_errors, train_top3_errors, train_top5_errors, test_losses, test_top1_errors, test_top3_errors, test_top5_errors = train(
        num_epochs=config["training"]["epochs"],
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        target_top1_error= config["training"]["target_top1_error"],
        target_top3_error=config["training"]["target_top3_error"],
        target_top5_error= config["training"]["target_top5_error"],
        device = config["device"],
        update_freq=config["training"]["update_freq"]
    )
    
    # Save weights
    os.makedirs("experiments/alexnet", exist_ok=True)
    torch.save(model.state_dict(), "experiments/alexnet/alexnet_weights.pth")

    # Save full checkpoint
    torch.save({
        'epoch': config["training"]["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_top1_errors': train_top1_errors,
        'train_top3_errors': train_top3_errors,
        'train_top5_errors': train_top5_errors,
        'test_losses': test_losses,
        'test_top1_errors': test_top1_errors,
        'test_top3_errors': test_top3_errors,
        'test_top5_errors': test_top5_errors,
    }, "experiments/alexnet/alexnet_detailed.pth")
    
    plot_loss(train_losses=train_losses, test_losses=test_losses, save_path="experiments/alexnet/loss.png")
    plot_accuracy(train_accuracies=train_top1_errors, test_accuracies=test_top1_errors, save_path="experiments/alexnet/top1_errors.png")
    plot_accuracy(train_accuracies=train_top3_errors, test_accuracies=test_top3_errors, save_path="experiments/alexnet/top3_errors.png")
    plot_accuracy(train_accuracies=train_top5_errors, test_accuracies=test_top5_errors, save_path="experiments/alexnet/top5_errors.png")
    plot_accuracy_improvement(test_top1_errors, save_path="experiments/alexnet/top1_error_improvement.png")
    plot_accuracy_improvement(test_top3_errors, save_path="experiments/alexnet/top3_error_improvement.png")
    plot_accuracy_improvement(test_top5_errors, save_path="experiments/alexnet/top5_error_improvement.png")
    
