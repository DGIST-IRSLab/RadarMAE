import torch.nn as nn

from models import backbone_2d, head

class RadarBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg_model = cfg.MODEL
        
        self.list_module_names = ['backbone', 'head']
        self.list_modules = []
        self.build_radar_detector()

    def build_radar_detector(self):
        for name_module in self.list_module_names:
            module = getattr(self, f'build_{name_module}')()
            self.add_module(name_module, module) # override nn.Module
            self.list_modules.append(module)

    def build_backbone(self):
        return backbone_2d.__all__[self.cfg_model.BACKBONE.NAME](self.cfg)

    def build_head(self):
        return head.__all__[self.cfg_model.HEAD.NAME](self.cfg)

    def forward(self, x):
        for module in self.list_modules:
            x = module(x)
        return x
