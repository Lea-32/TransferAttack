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
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --attack ma --model resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/ma/resnet50 --eval

    Notes:
        Download checkpoints ('aligned_res50.pt') from https://github.com/averyma/model-alignment, 
        and put them in the path '/path/to/checkpoints/'.
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
        if model_name in models.__dict__.keys() and model_name == 'resnet50':
            print('=> Loading model {} from torchvision.models'.format(model_name))
            model = models.get_model(model_name)
            ckpt_name = os.path.join(self.checkpoint_path, 'final_model.pt')

            # 加载权重
            ckpt = torch.load(ckpt_name, map_location='cpu')

            # 自动尝试移除常见的前缀（model. 或 module.）
            try:
                model.load_state_dict(ckpt)
            except RuntimeError:
                # 如果直接加载失败，尝试移除 "model." 前缀
                print("=> Attempting to load by removing 'model.' prefix...")
                new_ckpt = self.remove_prefix(ckpt, 'model.')
                try:
                    model.load_state_dict(new_ckpt)
                except RuntimeError:
                    # 如果还失败，尝试移除 "module." 前缀
                    print("=> Attempting to load by removing 'module.' prefix...")
                    new_ckpt = self.remove_prefix(ckpt, 'module.')
                    model.load_state_dict(new_ckpt)

        else:
            raise ValueError('Model {} not supported'.format(model_name))
        return wrap_model(model.eval().cuda())