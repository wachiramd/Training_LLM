import copy
import torch
import torch.nn as nn

from transformer.model import GPTLanguageModel


class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, rank: int, alpha: float) -> None:
        super().__init__()
        std_dev = 1/torch.sqrt(torch.tensor(rank).float())
        self.A = nn.Parameter(torch.randn(in_dim, rank)*std_dev)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.alpha*(x@self.A@self.B)
        return x


class LinearWithLoRA(nn.Module):
    def __init__(self, linear: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.linear = linear
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def print_trainable_parameters(model: GPTLanguageModel) -> None:
    trainable_parameters = 0
    all_parameters = 0
    for _, param in model.named_parameters():
        all_parameters += param.numel()
        if param.requires_grad:
            trainable_parameters += param.numel()

    print(
        f"All parameters: {all_parameters/1e6:.2f}M | "
        f"Trainable parameters: {trainable_parameters/1e6:.2f}M | "
        f"Trainable %: {100 * trainable_parameters / all_parameters:.2f}%"
    )


def get_lora_model(model: GPTLanguageModel, lora_config: dict, device: str) -> GPTLanguageModel:
    lora_model = copy.deepcopy(model)
    _replace_linear_layers_with_lora_layers(lora_model, lora_config)
    _freeze_non_lora_layers(lora_model)
    lora_model = lora_model.to(device)
    return lora_model


def _replace_linear_layers_with_lora_layers(module: nn.Module, lora_config: dict) -> None:
    rank = lora_config.get('rank', 4)
    alpha = lora_config.get('alpha', 8)

    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LinearWithLoRA(
                child, rank=rank, alpha=alpha))
        else:
            _replace_linear_layers_with_lora_layers(
                child, lora_config)


def _freeze_non_lora_layers(model: GPTLanguageModel) -> None:
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
