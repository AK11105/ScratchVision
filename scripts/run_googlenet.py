if __name__ == "__main__":
    import os
    import torch
    import torch.nn as nn
    import yaml

    from src.models.GoogLeNet import GoogLeNet
    from src.trainers.googlenet.train_googlenet import train
    from data.download_datasets import download_Imagenette
    from src.utils.loaders import create_DataLoaders
    from src.utils.visualization import *

    with open("configs/googlenet.yaml", "r") as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    model_cfg = config["model"]
    opt_cfg = train_cfg["optimizer"]

    train_dataset, test_dataset = download_Imagenette()
    train_dataloader, test_dataloader = create_DataLoaders(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle_train=train_cfg["shuffle_train"],
        num_workers=train_cfg["num_workers"]
    )

    model = GoogLeNet(output_dim=model_cfg["output_dim"], training=model_cfg["training"])

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=opt_cfg["lr"],
        momentum=opt_cfg["momentum"],
        weight_decay=opt_cfg["weight_decay"],
    )

    train_losses, train_top1_errors, train_top3_errors, train_top5_errors, \
    test_losses, test_top1_errors, test_top3_errors, test_top5_errors = train(
        num_epochs=train_cfg["epochs"],
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=config["device"],
        update_freq=train_cfg["update_freq"]
    )

    os.makedirs("experiments/googlenet", exist_ok=True)
    torch.save(model.state_dict(), "experiments/googlenet/googlenet_weights.pth")
    torch.save({
        'epoch': train_cfg["epochs"],
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
    }, "experiments/googlenet/googlenet_detailed.pth")

    plot_loss(train_losses=train_losses, test_losses=test_losses, save_path="experiments/googlenet/loss.png")
    plot_accuracy(train_accuracies=train_top1_errors, test_accuracies=test_top1_errors, save_path="experiments/googlenet/top1_errors.png")
    plot_accuracy(train_accuracies=train_top3_errors, test_accuracies=test_top3_errors, save_path="experiments/googlenet/top3_errors.png")
    plot_accuracy(train_accuracies=train_top5_errors, test_accuracies=test_top5_errors, save_path="experiments/googlenet/top5_errors.png")
    plot_accuracy_improvement(test_top1_errors, save_path="experiments/googlenet/top1_error_improvement.png")
    plot_accuracy_improvement(test_top3_errors, save_path="experiments/googlenet/top3_error_improvement.png")
    plot_accuracy_improvement(test_top5_errors, save_path="experiments/googlenet/top5_error_improvement.png")
