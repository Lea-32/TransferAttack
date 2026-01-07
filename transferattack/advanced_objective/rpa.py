import torch 
from ..utils import *
from ..gradient.mifgsm import MIFGSM

class RPA(MIFGSM):
    """
    Random Patch Attack
    'Enhancing the Transferability of Adversarial Examples with Random Patch (IJCAI 2022)' (https://www.ijcai.org/proceedings/2022/0233.pdf)
    Reference to the source code (https://github.com/alwaysfoggy/RPA)

    Arguments:
        model_name (str): the name of surrogate model for attack.
        epsilon (float): the perturbation budget.
        alpha (float): the step size.
        epoch (int): the number of iterations.
        decay (float): the decay factor for momentum calculation.
        patch_prob (float): the keep probability of patch. Drop probability = 1 - keep probability.
        num_ens (int): the number of gradients to aggregate
        targeted (bool): targeted/untargeted attack.
        random_start (bool): whether using random initialization for delta.
        norm (str): the norm of perturbation, l2/linfty.
        loss (str): the loss function.
        device (torch.device): the device for data. If it is None, the device would be same as model
        feature_layer: feature layer to launch the attack

    Official arguments:
        epsilon=16/255, alpha=epsilon/epoch=1.6/255, epoch=10, decay=1., num_ens=60

    Example script:
        python main.py --input_dir ./path/to/data --output_dir adv_data/rpa/resnet50 --attack rpa --model=resnet50
        python main.py --input_dir ./path/to/data --output_dir adv_data/rpa/resnet50 --eval
    """
    '''
        RPA是MIFGSM的子类，因此继承了MIFGSM的大部分方法
    '''

    # 构造器
    def __init__(self, model_name, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1, patch_prob=0.7, num_ens=30,
                targeted=False, random_start=False, norm='linfty', loss='crossentropy', device=None, attack='RPA', feature_layer='layer1', **kwargs):
        super().__init__(model_name, epsilon, alpha, epoch, decay, targeted, random_start, norm, loss, device, attack, **kwargs)
        # 进行多少次随即掩码的前向-反向以聚合梯度
        self.num_ens = num_ens
        # 指定中间层
        self.feature_layer = self.find_layer(feature_layer)
        # 掩码的keep probability，drop probability = 1 - patch
        self.patch_prob = patch_prob

    # 查找指定特征层，并返回该层的引用
    def find_layer(self,layer_name):
        parser = layer_name.split(' ')
        m = self.model[1]
        for layer in parser:
            if layer not in m._modules.keys():
                print("Selected layer is not in Model")
                exit() 
            else:
                m = m._modules.get(layer)
        return m

    # 定义前向传播的hook
    def __forward_hook(self, model, input, output):
        global mid_output
        mid_output = output

    # 定义反向传播的hook
    def __backward_hook(self, model, input, output):
        global mid_grad
        mid_grad = output

    def get_loss(self, agg_grad, mid_fmap):
        return -(agg_grad * mid_fmap).sum() if self.targeted else (agg_grad * mid_fmap).sum()

    # 按照stride方式构造patch掩码，并在部分path上随机初始化
    def patch_by_strides(self, img_shape, patch_size):
        img_shape = (img_shape[0], img_shape[2], img_shape[3], img_shape[1])
        x_mask = torch.ones(img_shape)
        N0, H0, W0, C0 = x_mask.shape
        ph = H0 // patch_size[0]
        pw = W0 // patch_size[1]
        x = x_mask[:, :ph * patch_size[0], :pw * patch_size[1]]
        N, H, W, C = x.shape

        size = (N, ph, pw, patch_size[0], patch_size[1], C)
        strides = (x.stride(0), x.stride(1)*patch_size[0], x.stride(2)*patch_size[0], *(x.stride()[1:]))
        mask_patchs = torch.as_strided(x, size=size, stride=strides)

        #
        mask_len = mask_patchs.shape[1] * mask_patchs.shape[2] * mask_patchs.shape[-1]
        # 随机选择patch数量
        rand_num = int(mask_len * (1 - self.patch_prob))
        # 随机选择patch索引
        rand_list = torch.randperm(mask_len)[:rand_num]

        for i in range(mask_patchs.shape[1]):
            for j in range(mask_patchs.shape[2]):
                for k in range(mask_patchs.shape[-1]):
                    if i * mask_patchs.shape[2] * mask_patchs.shape[-1] + j * mask_patchs.shape[-1] + k in rand_list:
                        mask_patchs[:, i, j, :, :, k] = torch.rand(N, mask_patchs.shape[3], mask_patchs.shape[4])
        # 把修改后的patch view展平成原图的形状，并赋回x_mask的对应位置
        img2 = torch.permute(mask_patchs, dims=(0,1,3,2,4,5))
        img2 = torch.flatten(img2, start_dim=0, end_dim=2)
        img2 = torch.flatten(img2, start_dim=1, end_dim=2)        
        img2 = img2.reshape((N, H, W, C))
        x_mask[:, :ph*patch_size[0], :pw*patch_size[1]] = img2

        return torch.permute(x_mask, dims=(0,3,1,2)).to(self.device)

    # 生成随机掩码，对每个掩码执行前反向，聚合mid-layer的梯度
    def get_agg_grad(self, data, label):
        x = torch.zeros(data.size()).cuda()
        x.copy_(data).detach()
        x.requires_grad = True
        batch_shape = data.shape
        # hook拿到中间层的grad_output，存到全局变量mid_grad中
        h2 = self.feature_layer.register_full_backward_hook(self.__backward_hook)
        agg_grad = 0
        # 3种不同尺寸的掩码，每3次使用一种
        for l in range(self.num_ens):
            # Generate random patch mask
            if l % 3 == 0:
                mask1 = torch.bernoulli(torch.ones_like(data), p=self.patch_prob)
                mask2 = torch.rand_like(data)
                mask = torch.where(mask1==1, 1, mask2)
            elif l % 3 == 1:
                mask = self.patch_by_strides(batch_shape, (3, 3))
            elif l % 3 == 2:
                mask = self.patch_by_strides(batch_shape, (5, 5))
            else:
                mask = self.patch_by_strides(batch_shape, (7, 7))

            # Obtain the logits outputs
            output_random = self.model(x*mask)
            output_random = torch.softmax(output_random, 1)
            
            # Calculate the loss
            loss = 0
            for batch_i in range(data.shape[0]):
                loss += output_random[batch_i][label[batch_i]]

            # Clean the gradients
            self.model.zero_grad()

            # Calculate the gradient of feature map
            loss.backward() # 触发backward hook，mid_grad存到全局变量mid_grad中

            # Aggregate the gradients of feature map
            agg_grad += mid_grad[0].detach() # 把中间层输出的梯度存到agg_grad中

        # Obtain the aggregate gradient
        # 对每个样本把聚合后的梯度 L2 归一化
        for batch_i in range(data.shape[0]):
            agg_grad[batch_i] /= agg_grad[batch_i].norm(2)
            
        h2.remove()# 移除hook
        return agg_grad

    # 攻击主循环，用聚合梯度引导delta更新
    def forward(self, data, label, **kwargs):
        """
        RPA attack procedure

        Arguments:
            data: (N, C, H, W) tensor for input images
            labels: (N,) tensor for ground-truth labels if untargetd, otherwise targeted labels
        """
        if self.targeted:
            assert len(label) == 2
            label = label[1] # the second element is the targeted label tensor
        data = data.clone().detach().to(self.device)
        label = label.clone().detach().to(self.device)

        # Initialize adversarial perturbation
        delta = self.init_delta(data)

        # Add hook
        h = self.feature_layer.register_forward_hook(self.__forward_hook)

        # Get aggregate gradient
        agg_grad = self.get_agg_grad(data, label)

        momentum = 0
        for _ in range(self.epoch):
            # Obtain the output
            logits = self.get_logits(self.transform(data + delta))
            
            # Calculate the loss
            loss = self.get_loss(agg_grad, mid_output)

            self.model.zero_grad()

            # Calculate the gradients
            grad = self.get_grad(loss, delta)

            # Calculate the momentum
            momentum = self.get_momentum(grad, momentum)

            # Update adversarial perturbation
            delta = self.update_delta(delta, data, -momentum, self.alpha)

        h.remove()
        return delta.detach()



