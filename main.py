import torch
from utils import read_configs
from model import VGG


def main():
    vgg_configs = read_configs('configs.json')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config_keys = list(vgg_configs.keys())
    for i, config in enumerate(config_keys):
        model = VGG(
            config=vgg_configs[config],
            in_channels=3,
            num_classes=1000
        ).to(device)
        print(f"Config: {config}, {model}")
        # total parameters in the model
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[INFO]: {total_params:,} total parameters. \n")


if __name__ == '__main__':
    main()
