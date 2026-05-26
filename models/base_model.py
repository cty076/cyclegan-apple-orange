import os
import torch
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from abc import ABC, abstractmethod
from . import networks


class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this function, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         define networks used in our training.
            -- self.visual_names (str list):        specify the images that you want to display and save.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        # 是否处于训练模式。训练时会创建 optimizer/scheduler，测试时通常只加载生成器。
        self.isTrain = opt.isTrain
        # 当前实验的 checkpoint 保存目录：checkpoints_dir/name。
        self.save_dir = Path(opt.checkpoints_dir) / opt.name  # save all the checkpoints to save_dir
        # 训练设备，来自 util.util.init_ddp()，可能是 CPU、单卡 GPU 或 DDP 下的本地 GPU。
        self.device = opt.device
        # with [scale_width], input images might have different sizes, which hurts the performance of cudnn.benchmark.
        if opt.preprocess != "scale_width":
            # 输入尺寸固定时打开 cudnn.benchmark，可以让 cuDNN 自动选择更快的卷积实现。
            torch.backends.cudnn.benchmark = True
        # 子类会填充下面这些列表。BaseModel 的打印、保存、可视化逻辑都依赖这些名字。
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.image_paths = []
        self.metric = 0  # used for learning rate policy 'plateau'

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    @abstractmethod
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # Initialize all networks and load if needed
        for name in self.model_names:
            if isinstance(name, str):
                # 例如 name="G_A" 时，取 self.netG_A。
                net = getattr(self, "net" + name)
                # 初始化网络权重。这里返回的是同一个网络对象，但可能已做初始化处理。
                net = networks.init_net(net, opt.init_type, opt.init_gain)

                # Load networks if needed
                if not self.isTrain or opt.continue_train:
                    # 测试或继续训练时，从磁盘读取已有权重。
                    # opt.load_iter > 0 时按 iter_xxx 命名，否则按 epoch/latest 命名。
                    load_suffix = f"iter_{opt.load_iter}" if opt.load_iter > 0 else opt.epoch
                    load_filename = f"{load_suffix}_net_{name}.pth"
                    load_path = self.save_dir / load_filename

                    # DDP 包装后的真实模型在 .module 里，加载权重前要取出来。
                    if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                        net = net.module
                    print(f"loading the model from {load_path}")

                    state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)

                    if hasattr(state_dict, "_metadata"):
                        del state_dict._metadata

                    # patch InstanceNorm checkpoints
                    # 兼容旧版 PyTorch 的 InstanceNorm checkpoint 字段差异。
                    for key in list(state_dict.keys()):
                        self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                    net.load_state_dict(state_dict)

                # Move network to device
                # 权重初始化/加载后，把网络放到目标设备上。
                net.to(self.device)

                # Wrap networks with DDP after loading
                if dist.is_initialized():
                    # Check if using syncbatch normalization for DDP
                    if self.opt.norm == "syncbatch":
                        raise ValueError(f"For distributed training, opt.norm must be 'syncbatch' or 'inst', but got '{self.opt.norm}'. " "Please set --norm syncbatch for multi-GPU training.")

                    # 分布式训练时用 DistributedDataParallel 包装网络。
                    net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[self.device.index])
                    # Sync all processes after DDP wrapping
                    dist.barrier()

                # 把可能已初始化、加载、迁移设备、DDP 包装后的网络写回 self.netX。
                setattr(self, "net" + name, net)

        self.print_networks(opt.verbose)

        if self.isTrain:
            # 为每个 optimizer 创建一个学习率调度器，后续每个 epoch 结束时更新学习率。
            self.schedulers = [networks.get_scheduler(optimizer, opt) for optimizer in self.optimizers]

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                # eval() 会关闭 Dropout 的随机性，并让 BatchNorm 使用统计均值方差。
                net.eval()

    def test(self):
        """Forward function used in test time.

        This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
        It also calls <compute_visuals> to produce additional visualization results
        """
        # 测试/推理不需要梯度，no_grad 可以省显存并加快速度。
        with torch.no_grad():
            self.forward()
            self.compute_visuals()

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        pass

    def get_image_paths(self):
        """Return image paths that are used to load current data"""
        return self.image_paths

    def update_learning_rate(self):
        """Update learning rates for all the networks; called at the end of every epoch"""
        old_lr = self.optimizers[0].param_groups[0]["lr"]
        for scheduler in self.schedulers:
            if self.opt.lr_policy == "plateau":
                # plateau 策略需要一个监控指标，这里使用 self.metric。
                scheduler.step(self.metric)
            else:
                scheduler.step()

        lr = self.optimizers[0].param_groups[0]["lr"]
        print(f"learning rate {old_lr:.7f} -> {lr:.7f}")

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        for name in self.visual_names:
            if isinstance(name, str):
                # 例如 visual_names 里有 "fake_B"，这里就取 self.fake_B。
                visual_ret[name] = getattr(self, name)
        return visual_ret

    def get_current_losses(self):
        """Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                # 例如 loss_names 里有 "cycle_A"，这里就取 self.loss_cycle_A。
                errors_ret[name] = float(getattr(self, "loss_" + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def save_networks(self, epoch):
        """Save all the networks to the disk, unwrapping them first."""

        # Only allow the main process (rank 0) to save the checkpoint
        if not dist.is_initialized() or dist.get_rank() == 0:
            for name in self.model_names:
                if isinstance(name, str):
                    # 文件名格式如 latest_net_G_A.pth 或 50_net_D_B.pth。
                    save_filename = f"{epoch}_net_{name}.pth"
                    save_path = self.save_dir / save_filename
                    net = getattr(self, "net" + name)

                    # 1. First, unwrap from DDP if it exists
                    # DDP 包装对象不能直接作为干净 checkpoint 保存，要取内部 module。
                    if hasattr(net, "module"):
                        model_to_save = net.module
                    else:
                        model_to_save = net

                    # 2. Second, unwrap from torch.compile if it exists
                    # torch.compile 包装后的原始模型在 _orig_mod 中。
                    if hasattr(model_to_save, "_orig_mod"):
                        model_to_save = model_to_save._orig_mod

                    # 3. Save the final, clean state_dict
                    # 只保存参数字典，不保存整个 Python 对象，便于跨环境加载。
                    torch.save(model_to_save.state_dict(), save_path)

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            # 旧版 InstanceNorm 可能保存了 running_mean/running_var；
            # 如果当前模块没有这些 buffer，就从 checkpoint 里删除。
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "running_mean" or key == "running_var"):
                if getattr(module, key) is None:
                    state_dict.pop(".".join(keys))
            if module.__class__.__name__.startswith("InstanceNorm") and (key == "num_batches_tracked"):
                state_dict.pop(".".join(keys))
        else:
            # 递归深入子模块，直到定位到具体参数/缓冲区名称。
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all networks from the disk for DDP."""

        for name in self.model_names:
            if isinstance(name, str):
                # 和 save_networks 的命名规则保持一致。
                load_filename = f"{epoch}_net_{name}.pth"
                load_path = self.save_dir / load_filename
                net = getattr(self, "net" + name)

                # DDP 包装时真实模型在 .module 中。
                if isinstance(net, torch.nn.parallel.DistributedDataParallel):
                    net = net.module
                print(f"loading the model from {load_path}")

                state_dict = torch.load(load_path, map_location=str(self.device), weights_only=True)

                if hasattr(state_dict, "_metadata"):
                    del state_dict._metadata

                # patch InstanceNorm checkpoints
                # 加载前清理旧版 InstanceNorm 中当前模型不需要的字段。
                for key in list(state_dict.keys()):
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split("."))
                net.load_state_dict(state_dict)

        # Add a barrier to sync all processes before continuing
        if dist.is_initialized():
            dist.barrier()

    def print_networks(self, verbose):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        print("---------- Networks initialized -------------")
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)
                num_params = 0
                # 统计所有参数元素个数，用百万参数 M 为单位打印。
                for param in net.parameters():
                    num_params += param.numel()
                if verbose:
                    print(net)
                print(f"[Network {name}] Total number of parameters : {num_params / 1e6:.3f} M")
        print("-----------------------------------------------")

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    # 关闭 requires_grad 后，反向传播不会为这些参数计算梯度。
                    # CycleGAN 更新 G 时会冻结 D，更新 D 时再解冻。
                    param.requires_grad = requires_grad

    def init_networks(self, init_type="normal", init_gain=0.02):
        """Initialize all networks: 1. move to device; 2. initialize weights

        Parameters:
            init_type (str) -- initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float) -- scaling factor for normal, xavier and orthogonal
        """
        import os

        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, "net" + name)

                # Move to device
                # 这个方法是较旧/备用初始化入口；setup() 里也会做类似设备迁移和初始化。
                if torch.cuda.is_available():
                    if "LOCAL_RANK" in os.environ:
                        local_rank = int(os.environ["LOCAL_RANK"])
                        net.to(local_rank)
                        print(f"Initialized network {name} with device cuda:{local_rank}")
                    else:
                        net.to(0)
                        print(f"Initialized network {name} with device cuda:0")
                else:
                    net.to("cpu")
                    print(f"Initialized network {name} with device cpu")

                # Initialize weights using networks function
                networks.init_weights(net, init_type, init_gain)
