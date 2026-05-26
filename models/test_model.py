from .base_model import BaseModel
from . import networks


class TestModel(BaseModel):
    """This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        # TestModel 只用于推理，不允许训练。
        assert not is_train, "TestModel cannot be used during training time"
        # single 数据集模式只从一个文件夹读图片，不需要 A/B 两个域同时输入。
        parser.set_defaults(dataset_mode="single")
        # 用 model_suffix 决定加载哪个生成器权重，例如 _A 或 _B。
        parser.add_argument("--model_suffix", type=str, default="", help="In checkpoints_dir, [epoch]_net_G[model_suffix].pth will be loaded as the generator.")

        return parser

    def __init__(self, opt):
        """Initialize the single-direction CycleGAN test model.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        # 这个模型只用于测试，因此 opt.isTrain 必须为 False。
        assert not opt.isTrain
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        # 推理没有训练损失，所以列表为空。
        self.loss_names = []
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        # 测试时只展示输入 real 和生成结果 fake。
        self.visual_names = ["real", "fake"]
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # 只需要一个生成器。model_suffix 让 BaseModel 能按文件名加载对应权重。
        self.model_names = ["G" + opt.model_suffix]  # only generator is needed.
        # 创建单向生成器，例如 A->B 或 B->A。
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        # assigns the model to self.netG_[suffix] so that it can be loaded
        # please see <BaseModel.load_networks>
        # BaseModel.load_networks 会查找 self.netG + suffix，因此这里动态设置属性。
        setattr(self, "netG" + opt.model_suffix, self.netG)  # store netG in self.

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input: a dictionary that contains the data itself and its metadata information.

        We need to use 'single_dataset' dataset mode. It only load images from one domain.
        """
        # single_dataset 固定把输入图片放在 input["A"] 中。
        self.real = input["A"].to(self.device)
        # 保存路径用于生成输出文件名和 HTML 展示。
        self.image_paths = input["A_paths"]

    def forward(self):
        """Run forward pass."""
        # 推理时只执行一次单向生成：real -> fake。
        self.fake = self.netG(self.real)  # G(real)

    def optimize_parameters(self):
        """No optimization for test model."""
        # 测试模型不训练参数，因此这里留空。
        pass
