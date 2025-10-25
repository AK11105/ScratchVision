import torch

from ..components.neocognitron import SLayer
from ..components.neocognitron import CLayer
from ..components.neocognitron import patch_extract

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Neocognitron:
    def __init__(self, config: dict, device=Device):
        self.device = device
        self.layers_cfg = config['layers']
        self.layers = []
        self.slayers = []
        in_ch = 1
        in_map = config.get('input_size', (16, 16))
        
        for cfg in self.layers_cfg: #Loop through configuration layers
            if cfg['type'] == 'S': #Learn Features
                s = SLayer(in_ch, cfg['planes'], cfg['rf'], in_map, cfg['out_map'], device)
                s.r = cfg.get('r', 1.0)
                self.layers.append(('S', s, cfg))
                self.slayers.append(s)
                in_ch = cfg['planes']
                in_map = cfg['out_map']
            else: #Generalize Features
                c = CLayer(in_ch, cfg['planes'], cfg['rf'], in_map, cfg['out_map'],alpha=cfg.get('alpha', 1.0), device=device)
                self.layers.append(('C', c, cfg))
                in_ch = cfg['planes']
                in_map = cfg['out_map']
            
    def forward(self, x):
        acts = []
        cur = x
        for t, obj, cfg in self.layers: #Loop through layers
            if t == 'S': #Convolutional Layer
                # Adjust stride and padding
                stride = cfg.get('stride', 1)
                padding = cfg.get('padding', 0)
                
                # Make sure kernel fits the current input
                rf = min(cfg['rf'], cur.shape[-1], cur.shape[-2])
                if rf < cfg['rf']:
                    padding += (cfg['rf'] - rf) // 2
                out, patches = obj.forward(cur, stride=cfg.get('stride', 1), padding=cfg.get('padding', 0))
                acts.append(('S', out, patches, obj))
                cur = out
            else: #Pooling Layer
                stride = cfg.get('stride', 1)
                padding = cfg.get('padding', 0)
                rf = min(cfg['rf'], cur.shape[-1], cur.shape[-2])
                if rf < cfg['rf']:
                    padding += (cfg['rf'] - rf) // 2
                patches = patch_extract(cur, kernel=cfg['rf'], stride=cfg.get('stride', 1), padding=cfg.get('padding', 0))
                out = obj.forward(patches)
                acts.append(('C', out, patches, obj))
                cur = out
        return acts, cur