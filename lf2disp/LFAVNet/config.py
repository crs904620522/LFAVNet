import torch
import torch.distributions as dist
from torch import nn
import os
from lf2disp.LFAVNet import models, training, generation
from lf2disp.LFAVNet.datafield.HCInew_dataloader import HCInew

Datadict = {
    'HCInew':HCInew,
}

def get_model(cfg, dataset=None, device=None):
    model = models.LFAVNet(cfg, device=device)
    return model


def get_dataset(mode, cfg):
    type = cfg['data']['dataset']
    dataset = Datadict[type](cfg, mode=mode)
    return dataset


def get_trainer(model, optimizer, cfg, criterion, device, **kwargs):
    trainer = training.Trainer(
        model, optimizer,
        device=device,
        criterion=criterion,
        cfg=cfg,
    )
    return trainer


def get_generator(model, cfg, device, **kwargs):
    generator = generation.GeneratorDepth(
        model,
        device=device,
        cfg=cfg,
    )
    return generator
