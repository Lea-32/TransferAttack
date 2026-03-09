import torch

from ..utils import *
from ..attack import Attack

class MA(Attack):
    """
    MA Attack
    'Improving Adversarial Transferability via Model Alignment (ECCV 2024)'(https://arxiv.org/abs/2311.18495)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model.

    Official arguments:
        epsilon=4/255, alpha=1/255, epoch=20, decay=0.

    Example script:
        # ResNet50 (需 aligned_res50.pt)
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --attack ma --model resnet50
        # ViT (需 final_model.pt，含 model.xxx 格式的 ViT-Base 权重)
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/vit --attack ma --model vit_base_patch16_224
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --eval

    Notes:
        - ResNet50: 下载 aligned_res50.pt 放到 checkpoints/
        - ViT: 将 final_model.pt (ViT-Base 12 blocks) 放到 checkpoints/
    """

    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., targeted=False, random_start=False,
                norm='linfty', loss='crossentropy', device=None, attack='MA', checkpoint_path='./checkpoints', **kwargs):
        self.checkpoint_path = checkpoint_path
        super().__init__(attack, model_name, epsilon, targeted, random_start, norm, loss, device)
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay

    def remove_prefix(self, state_dict, prefix='model.'):
        """
        移除 state_dict 中键名的特定前缀
        """
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # 如果键名以指定的 prefix 开头，则截掉它
            if k.startswith(prefix):
                name = k[len(prefix):]
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict

    def load_model(self, model_name):
        # ViT 架构：从 final_model.pt 加载（checkpoint 含 model.xxx 格式的 ViT 权重）
        if model_name == 'vit_base_patch16_224':
            print('=> Loading ViT from final_model.pt')
            model = timm.create_model('vit_base_patch16_224', pretrained=False)
            ckpt_name = os.path.join(self.checkpoint_path, 'final_model.pt')
            if not os.path.exists(ckpt_name):
                raise FileNotFoundError(f"Checkpoint 不存在: {ckpt_name}")
            ckpt = torch.load(ckpt_name, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                ckpt = ckpt['model']
            # 移除 model. 前缀以匹配 timm 的 ViT 结构
            if any(k.startswith('model.') for k in ckpt.keys()):
                ckpt = self.remove_prefix(ckpt, 'model.')
            model.load_state_dict(ckpt, strict=True)

        elif model_name in models.__dict__.keys() and model_name == 'resnet50':
            print('=> Loading model {} from torchvision.models'.format(model_name))
            model = models.get_model(model_name)
            ckpt_name = os.path.join(self.checkpoint_path, 'aligned_res50.pt')
            if not os.path.exists(ckpt_name):
                ckpt_name = os.path.join(self.checkpoint_path, 'final_model.pt')
            ckpt = torch.load(ckpt_name, map_location='cpu')
            if isinstance(ckpt, dict) and 'state_dict' in ckpt:
                ckpt = ckpt['state_dict']
            elif isinstance(ckpt, dict) and 'model' in ckpt:
                ckpt = ckpt['model']
            try:
                model.load_state_dict(ckpt)
            except RuntimeError:
                new_ckpt = self.remove_prefix(ckpt, 'model.')
                try:
                    model.load_state_dict(new_ckpt)
                except RuntimeError:
                    new_ckpt = self.remove_prefix(ckpt, 'module.')
                    model.load_state_dict(new_ckpt)

        else:
            raise ValueError('Model {} not supported. MA 支持: resnet50, vit_base_patch16_224'.format(model_name))
        return wrap_model(model.eval().cuda())