from doctest import master
from torchvision import transforms, datasets
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader, DistributedSampler
import torch
import resnet
import torch.optim as optim
import torch.nn as nn
import numpy as np
import torch_xla.distributed.parallel_loader as pl
import os
import argparse
from datetime import datetime
import torch_xla.distributed.xla_multiprocessing as xmp


def get_transform(dset):
    if dset == "cifar10":
        train_transform = [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        val_transform = []
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    else:
        raise NotImplementedError(f"args.dset = {dset} is not implemented")
    transform = {}
    transform["train"] = transforms.Compose([*train_transform, transforms.ToTensor(), normalize])
    transform["val"] = transforms.Compose([*val_transform, transforms.ToTensor(), normalize])
    return transform


def get_dataset(dset, root, transform):
    if dset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=root, train=True, transform=transform["train"], download=True
        )
        val_dataset = datasets.CIFAR10(
            root=root, train=False, transform=transform["val"], download=True
        )
        n_class = 10
    else:
        raise NotImplementedError(f"args.dset = {dset} is not implemented")
    dataset = {}
    dataset["train"] = train_dataset
    dataset["val"] = val_dataset
    dataset["n_class"] = n_class
    return dataset


def get_loader(dataset, batch_size, n_workers):
    sampler = {"train": None, "val": None}
    if xm.xrt_world_size() > 1:
        sampler["train"] = DistributedSampler(
            dataset["train"], num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=True
        )
        sampler["val"] = DistributedSampler(
            dataset["val"], num_replicas=xm.xrt_world_size(), rank=xm.get_ordinal(), shuffle=False
        )
    loader = {}
    loader["train"] = DataLoader(
        dataset["train"],
        batch_size=batch_size,
        sampler=sampler["train"],
        drop_last=True,
        shuffle=False if sampler["train"] else True,
        num_workers=n_workers,
    )
    loader["val"] = DataLoader(
        dataset["val"],
        batch_size=batch_size,
        sampler=sampler["val"],
        drop_last=False,
        shuffle=False if sampler["val"] else True,
        num_workers=n_workers,
    )
    return loader


def get_model(net, n_class):
    if net == "resnet18":
        model = resnet.ResNet18(n_class=n_class)
    elif net == "resnet34":
        model = resnet.ResNet34(n_class=n_class)
    elif net == "resnet50":
        model = resnet.ResNet50(n_class=n_class)
    elif net == "resnet101":
        model = resnet.ResNet101(n_class=n_class)
    elif net == "resnet152":
        model = resnet.ResNet152(n_class=n_class)
    else:
        raise NotImplementedError(f"args.net = {net} is not implemented")
    return model


def get_optimizer(opt, model, lr):
    if opt == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = None
    return optimizer


def get_lr_scheduler(lr_sched, optimizer, n_train_steps):
    if lr_sched == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_train_steps)
    else:
        raise NotImplementedError(f"args.lr_sched = {lr_sched} is not implemented")
    return lr_scheduler


class RunningMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        attr_keys = list(self.__dict__.keys())
        for key in attr_keys:
            delattr(self, key)
        self.cntr = 0

    def add(self, **kwargs):
        for key, value in kwargs.items():
            attr = getattr(self, key, None)
            if attr is None and self.cntr == 0:
                setattr(self, key, value)
            elif attr is None:
                raise ValueError(f"invalid key: {key}")
            else:
                attr = attr + value
                setattr(self, key, attr)
        self.cntr += 1

    def get(self, reduce=False):
        metrics = {
            key: value.item() / self.cntr for key, value in self.__dict__.items() if key != "cntr"
        }
        if reduce:
            metrics = {key: xm.mesh_reduce(key, value, np.mean) for key, value in metrics.items()}
        return metrics

    def msg(self, avg_metrics=None):
        if avg_metrics == None:
            avg_metrics = self.get()
        return "".join(["[{}] {:.5f} ".format(key, value) for key, value in avg_metrics.items()])

    def print(self, progress=0, bar_len=20, msg=None):
        msg = self.msg() if msg == None else msg
        msg = msg.ljust(50)
        block = int(round(bar_len * progress))
        text = "\rprogress: [{}] {}% {}".format(
            "\x1b[32m" + "=" * (block - 1) + ">" + "\033[0m" + "-" * (bar_len - block),
            round(progress * 100, 2),
            msg,
        )
        print(text, end="")
        if progress == 1:
            print()


def train(args):
    transform = get_transform(args.dset)
    dataset = get_dataset(args.dset, args.data_root, transform)
    loader = get_loader(dataset, args.batch_size, args.n_workers)

    torch.manual_seed(args.seed)
    device = xm.xla_device()
    model = get_model(args.net, dataset["n_class"]).to(device)
    optimizer = get_optimizer(args.opt, model, args.lr)
    lr_scheduler = get_lr_scheduler(args.lr_sched, optimizer, args.n_epochs)
    criterion = nn.CrossEntropyLoss()
    tracker = RunningMeter()
    master_ordinal = xm.is_master_ordinal()

    def train_epoch(train_loader):
        model.train()
        tracker.reset()
        for indx, (img, target) in enumerate(train_loader):
            output = model(img)

            loss = criterion(output, target)
            pred_class = output.argmax(dim=1)
            acc = pred_class.eq(target.view_as(pred_class)).sum() / img.shape[0]

            optimizer.zero_grad()
            loss.backward()
            xm.optimizer_step(optimizer)

            tracker.add(train_loss=loss, train_acc=acc)
            if master_ordinal and indx % args.log_freq == 0:
                tracker.print(indx/len(train_loader))
        if master_ordinal:
            tracker.print(1)

    def val(val_loader):
        model.eval()
        tracker.reset()
        for indx, (img, target) in enumerate(val_loader):
            output = model(img)

            loss = criterion(output, target)
            pred_class = output.argmax(dim=1)
            acc = pred_class.eq(target.view_as(pred_class)).sum() / img.shape[0]

            tracker.add(val_loss=loss, val_acc=acc)
            if master_ordinal and indx % args.log_freq == 0:
                tracker.print(indx/len(val_loader), msg="")
        metrics = tracker.get(True)
        if master_ordinal:
            tracker.print(1, msg=tracker.msg(metrics))
        return metrics

    train_device_loader = pl.MpDeviceLoader(loader["train"], device)
    val_device_loader = pl.MpDeviceLoader(loader["val"], device)
    best_val = 0
    for epoch in range(args.n_epochs):
        if master_ordinal:
            print(f"epoch: {epoch}")
            print("---------------")
        train_epoch(train_device_loader)
        if epoch % args.val_freq == 0 or epoch == args.n_epochs - 1:
            metrics = val(val_device_loader)
            if master_ordinal and metrics["val_acc"] > best_val:
                print(
                    "\x1b[33m"
                    + f"val acc improved from {round(best_val, 5)} to {round(metrics['val_acc'], 5)}"
                    + "\033[0m"
                )
                torch.save(
                    model.state_dict(),
                    os.path.join(args.out_dir, f"best.ckpt"),
                )
            best_val = metrics["val_acc"]
        if master_ordinal:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler else None,
                    "epoch": epoch,
                },
                os.path.join(args.out_dir, "last.ckpt"),
            )


def _mp_fn(index, args):
    torch.set_default_tensor_type("torch.FloatTensor")
    train(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_cores", type=int, default=1, help="set number of tpu cores [d: 1]")
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"out/{datetime.now().strftime('%Y-%m-%d_%H-%M')}",
        help="path to output directory [d: out/year-month-date_hour-minute]",
    )
    parser.add_argument("--seed", type=int, default=42, help="set experiment seed [d: 42]")
    parser.add_argument("--dset", type=str, default="cifar10", help="dataset name [d: cifar10]")
    parser.add_argument("--data_root", type=str, required=True, help="dataset directory")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size [d: 128]")
    parser.add_argument(
        "--n_workers", type=int, default=4, help="number of workers for dataloading [d: 4]"
    )
    parser.add_argument("--net", type=str, default="resnet18", help="network name [d: resnet18]")
    parser.add_argument("--opt", type=str, default="sgd", help="optimizer name [d: sgd]")
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate [d: 0.1]")
    parser.add_argument(
        "--n_epochs", type=int, default=200, help="number of training epochs [d: 200]"
    )
    parser.add_argument(
        "--lr_sched", type=str, default="cosine", help="lr scheduler name [d: cosine]"
    )
    parser.add_argument("--val_freq", type=int, default=10, help="validation frequency [d: 10]")
    parser.add_argument("--log_freq", type=int, default=100, help="logging frequency [d: 100]")
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    xmp.spawn(_mp_fn, args=(args,), nprocs=args.n_cores)
