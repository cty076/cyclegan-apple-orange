import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    """
    This class implements the CycleGAN model, for learning image-to-image translation without paired data.

    The model training requires '--dataset_mode unaligned' dataset.
    By default, it uses a '--netG resnet_9blocks' ResNet generator,
    a '--netD basic' PatchGAN discriminator,
    and a least-square GANs objective ('--gan_mode lsgan').

    CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For CycleGAN, in addition to GAN losses, we introduce lambda_A, lambda_B, and lambda_identity for the following losses.
        A (source domain), B (target domain).
        Generators: G_A: A -> B; G_B: B -> A.
        Discriminators: D_A: G_A(A) vs. B; D_B: G_B(B) vs. A.
        Forward cycle loss:  lambda_A * ||G_B(G_A(A)) - A|| (Eqn. (2) in the paper)
        Backward cycle loss: lambda_B * ||G_A(G_B(B)) - B|| (Eqn. (2) in the paper)
        Identity loss (optional): lambda_identity * (||G_A(B) - B|| * lambda_B + ||G_B(A) - A|| * lambda_A) (Sec 5.2 "Photo generation from paintings" in the paper)
        Dropout is not used in the original CycleGAN paper.
        """
        # CycleGAN 论文原版生成器默认不使用 dropout，因此这里覆盖通用默认值。
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        if is_train:
            # A -> B -> A 这条循环重建损失的权重。
            parser.add_argument("--lambda_A", type=float, default=10.0, help="weight for cycle loss (A -> B -> A)")
            # B -> A -> B 这条循环重建损失的权重。
            parser.add_argument("--lambda_B", type=float, default=10.0, help="weight for cycle loss (B -> A -> B)")
            parser.add_argument(
                "--lambda_identity",
                type=float,
                default=0.5,
                help="use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1",
            )

        return parser

    def __init__(self, opt):
        """Initialize the CycleGAN class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        # 这些名字会映射到 self.loss_D_A、self.loss_G_A 等变量，用于日志打印和 loss 曲线。
        self.loss_names = ["D_A", "G_A", "cycle_A", "idt_A", "D_B", "G_B", "cycle_B", "idt_B"]
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        # A 方向可视化：真实 A、A->B 生成图、A->B->A 重建图。
        visual_names_A = ["real_A", "fake_B", "rec_A"]
        # B 方向可视化：真实 B、B->A 生成图、B->A->B 重建图。
        visual_names_B = ["real_B", "fake_A", "rec_B"]
        if self.isTrain and self.opt.lambda_identity > 0.0:  # if identity loss is used, we also visualize idt_B=G_A(B) ad idt_A=G_B(A)
            # identity 可视化：目标域输入生成器后应尽量保持不变。
            visual_names_A.append("idt_B")
            visual_names_B.append("idt_A")

        self.visual_names = visual_names_A + visual_names_B  # combine visualizations for A and B
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>.
        if self.isTrain:
            # 训练时四个网络都要保存：两个生成器 + 两个判别器。
            self.model_names = ["G_A", "G_B", "D_A", "D_B"]
        else:  # during test time, only load Gs
            # 测试/推理只需要生成器，不需要判别器。
            self.model_names = ["G_A", "G_B"]

        # define networks (both Generators and discriminators)
        # The naming is different from those used in the paper.
        # Code (vs. paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)
        # G_A 学习 A -> B；输入通道 opt.input_nc，输出通道 opt.output_nc。
        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)
        # G_B 学习 B -> A；输入/输出通道和 G_A 相反。
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm, not opt.no_dropout, opt.init_type, opt.init_gain)

        if self.isTrain:  # define discriminators
            # D_A 判断 B 域图片真假：real_B 为真，G_A(real_A)=fake_B 为假。
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)
            # D_B 判断 A 域图片真假：real_A 为真，G_B(real_B)=fake_A 为假。
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain)

        if self.isTrain:
            if opt.lambda_identity > 0.0:  # only works when input and output images have the same number of channels
                # identity loss 要直接比较 G_A(B) 与 B、G_B(A) 与 A，因此输入输出通道必须一致。
                assert opt.input_nc == opt.output_nc
            # 历史 fake 图像池。训练判别器时混入历史生成图，能减少 GAN 振荡。
            self.fake_A_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            self.fake_B_pool = ImagePool(opt.pool_size)  # create image buffer to store previously generated images
            # define loss functions
            # 对抗损失，默认 lsgan：真实标签为 1，假标签为 0。
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            # cycle consistency loss 使用 L1：重建图 rec_A/rec_B 要接近原图 real_A/real_B。
            self.criterionCycle = torch.nn.L1Loss()
            # identity loss 也使用 L1：目标域图片输入对应生成器后应基本不变。
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # 两个生成器一起更新，因此把 G_A 和 G_B 的参数串到同一个 Adam optimizer。
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # 两个判别器一起更新，因此把 D_A 和 D_B 的参数串到同一个 Adam optimizer。
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            # BaseModel.setup() 会遍历 self.optimizers，为它们创建学习率调度器。
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        # direction=AtoB 时保持数据集 A/B 原样；direction=BtoA 时交换 A/B。
        AtoB = self.opt.direction == "AtoB"
        # real_A 和 real_B 是当前 batch 的两个域图片。CycleGAN 不要求它们配对。
        self.real_A = input["A" if AtoB else "B"].to(self.device)
        self.real_B = input["B" if AtoB else "A"].to(self.device)
        # 保存输入路径，测试/可视化时用于命名输出结果。
        self.image_paths = input["A_paths" if AtoB else "B_paths"]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # A -> B：生成目标域 B 风格图像。
        self.fake_B = self.netG_A(self.real_A)  # G_A(A)
        # A -> B -> A：把 fake_B 再翻译回 A，用于 cycle consistency。
        self.rec_A = self.netG_B(self.fake_B)  # G_B(G_A(A))
        # B -> A：生成目标域 A 风格图像。
        self.fake_A = self.netG_B(self.real_B)  # G_B(B)
        # B -> A -> B：把 fake_A 再翻译回 B，用于 cycle consistency。
        self.rec_B = self.netG_A(self.fake_A)  # G_A(G_B(B))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        # 判别器看到真实图片时，目标标签是真。
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        # detach() 切断 fake 与生成器的计算图：当前只训练 D，不更新 G。
        pred_fake = netD(fake.detach())
        # 判别器看到生成图片时，目标标签是假。
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        # real/fake 两部分平均，避免 D 的梯度尺度过大。
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # 这里只计算判别器梯度；真正参数更新在 optimizer_D.step()。
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        # D_A 是 B 域判别器：real_B 为真，fake_B 为假。
        # fake_B_pool 可能返回当前 fake_B，也可能返回历史 fake_B。
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        # D_B 是 A 域判别器：real_A 为真，fake_A 为假。
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed: ||G_A(B) - B||
            # G_A 的目标域是 B；如果输入已经是 B，就应该尽量输出原图 B。
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed: ||G_B(A) - A||
            # G_B 的目标域是 A；如果输入已经是 A，就应该尽量输出原图 A。
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            # 不启用 identity loss 时，用 0 占位，便于后面统一求和。
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        # 更新生成器时，fake_B 的目标标签是真：希望 G_A 骗过 B 域判别器 D_A。
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        # GAN loss D_B(G_B(B))
        # 更新生成器时，fake_A 的目标标签是真：希望 G_B 骗过 A 域判别器 D_B。
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        # A -> B -> A 重建损失，约束翻译后仍保留 A 的内容结构。
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        # B -> A -> B 重建损失，约束翻译后仍保留 B 的内容结构。
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        # combined loss and calculate gradients
        # 生成器总损失 = 两个对抗损失 + 两个循环损失 + 可选 identity 损失。
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B
        # 这里只计算 G_A/G_B 的梯度；真正参数更新在 optimizer_G.step()。
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        # 先得到 fake_B、rec_A、fake_A、rec_B，后续 G/D 的 loss 都依赖这些结果。
        self.forward()  # compute fake images and reconstruction images.
        # G_A and G_B
        # 更新生成器时冻结判别器参数，但仍会通过判别器输出计算生成器的 GAN loss。
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()  # calculate gradients for G_A and G_B
        self.optimizer_G.step()  # update G_A and G_B's weights
        # D_A and D_B
        # 更新判别器时重新打开判别器梯度；fake 图在 backward_D_basic 中会 detach。
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()  # set D_A and D_B's gradients to zero
        self.backward_D_A()  # calculate gradients for D_A
        self.backward_D_B()  # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
