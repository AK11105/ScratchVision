if __name__ == "__main__":
    import torch 
    import yaml 
    import os 
    import torch.nn as nn 
    import torch.optim as optim 
    from src.models.LeNet import LeNet5
    from data.download_datasets import download_MNIST
    from src.utils.loaders import create_DataLoaders
    from src.utils.visualization import *
    from src.trainers import train_lenet
    from src.components.lenet import ScaledTanH
    from src.evaluators.metrics import DiscriminativeRBFLoss
    
    with open('configs/lenet.yaml', "r") as lenet:
        config = yaml.safe_load(lenet)
        
    batch_size = config["training"]["batch_size"]
    shuffle_train = config["training"]["shuffle_train"]
    num_workers = config["training"]["num_workers"]
    
    train_dataset, test_dataset = download_MNIST()
    train_dataloader, test_dataloader = create_DataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        shuffle_train=shuffle_train,
        num_workers=num_workers
    )
    
    layers_dict = {layer["type"]: layer for layer in config["layers"]}
    C1 = layers_dict.get("C1")
    S2 = layers_dict.get("S2")
    S4 = layers_dict.get("S4")
    C5 = layers_dict.get("C5")
    F6 = layers_dict.get("F6")
    RBF = layers_dict.get("RBF")
    
    model = LeNet5(
        activation=ScaledTanH() if config["model"]["activation"] == "ScaledTanH" else None,
        C1_in=config["model"]["input_channels"],
        C1_out=C1["planes"],
        C1_filter=C1["rf"],
        C1_stride=C1["stride"],
        C1_padding=C1.get("padding", 0),

        S2_filter=S2["rf"],
        S2_stride=S2["stride"],

        S4_filter=S4["rf"],
        S4_stride=S4["stride"],

        C5_in=S4["planes"],  # from previous layer output planes
        C5_out=C5["planes"], # usually equals planes
        C5_filter=C5["rf"],
        C5_stride=C5["stride"],
        C5_padding=C5.get("padding", 0),

        F6_in=C5["planes"],
        F6_out=F6["units"],

        output_in=F6["units"],
        output_out=RBF["units"]
    )
    
    criterion = DiscriminativeRBFLoss() if config["training"]["loss_function"] else nn.MSELoss()
    
    optimizer_cfg = config["training"]["optimizer"]
    optimizer_params = optimizer_cfg["params"]
    lr_schedule = config["training"]["lr_schedule"]

    optimizer = optim.SGD(
        model.parameters(),
        momentum=optimizer_params.get("momentum", 0),
        weight_decay=optimizer_params.get("weight_decay", 0),
        lr=0.0  # Initially set to 0; will be controlled by LR schedule
    )
    
    train_losses, train_accuracies, test_losses, test_accuracies = train_lenet(
        num_epochs=config["training"]["epochs"],
        model=model,
        train_loader=train_dataloader,
        test_loader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        target_accuracy=config["training"]["target_accuracy"],
        device = config["device"],
        update_freq=config["training"]["update_frequency"],
        lr_schedule=lr_schedule
    )
    
    # Save weights
    os.makedirs("experiments/lenet", exist_ok=True)
    torch.save(model.state_dict(), "experiments/lenet/lenet_weights.pth")

    # Save full checkpoint
    torch.save({
        'epoch': config["training"]["epochs"],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, "experiments/lenet/lenet_detailed.pth")
    
    plot_loss(train_losses=train_losses, test_losses=test_losses, save_path="experiments/lenet/loss.png")
    plot_accuracy(train_accuracies=train_accuracies, test_accuracies=test_accuracies, save_path="experiments/lenet/accuracy.png")
    plot_accuracy_improvement(test_accuracies, save_path="experiments/lenet/accuracy_improvement.png")
    
    
