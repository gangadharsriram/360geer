"""VGG16 perceptual loss: L1 between frozen-VGG16 features (relu1_2, relu2_2, relu3_3, relu4_3) of render and GT."""
import torch
import torch.nn.functional as F
from torchvision import models

_net = None
_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_TAPS = (3, 8, 15, 22)


def _get_net():
    global _net
    if _net is None:  # lazy: only built when lambda_vgg > 0; weights come from the torch hub cache
        _net = models.vgg16(pretrained=True).features[:_TAPS[-1] + 1].cuda().eval()
        for p in _net.parameters():
            p.requires_grad_(False)
    return _net


def vgg_loss(image, gt, crop=512):
    """image, gt: (3,H,W) or (1,3,H,W) in [0,1]. crop>0 compares one random crop x crop patch (full image if 0 / too big)."""
    image, gt = image.reshape(-1, *image.shape[-3:]), gt.reshape(-1, *gt.shape[-3:])
    H, W = image.shape[-2:]
    if 0 < crop < min(H, W):
        y, x = torch.randint(0, H - crop + 1, (1,)).item(), torch.randint(0, W - crop + 1, (1,)).item()
        image, gt = image[..., y:y + crop, x:x + crop], gt[..., y:y + crop, x:x + crop]
    net, loss = _get_net(), 0.0
    a, b = (image - _MEAN.to(image.device)) / _STD.to(image.device), (gt - _MEAN.to(gt.device)) / _STD.to(gt.device)
    for i, layer in enumerate(net):
        a, b = layer(a), layer(b)
        if i in _TAPS:
            loss = loss + F.l1_loss(a, b.detach())
    return loss
