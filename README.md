# 高动态范围成像

本项目主要目标是实现论文 "Deep High Dynamic Range Imaging of Dynamic Scenes" 中的方法并且进行优化和创新。由吴烨晗和唐凯两人共同完成。通过使用深度学习技术，我们学习在动态场景中进行高动态范围成像：

<center>
<img src="assets/examples.jpg">
| （左）LDR 输入 | （中）从预测的 HDR 中提取的曝光 | （右）色调映射后的 HDR |
</center>

---

#### 实现以下论文和代码演示:

**ExpandNet**: 从低动态范围内容扩展到高动态范围的深度卷积神经网络

[Deep High Dynamic Range Imaging of Dynamic Scenes](https://people.engr.tamu.edu/nimak/Data/SIGGRAPH17_HDR_LoRes.pdf)

#### 代码演示(BILIBILI)

高动态范围成像（https://www.bilibili.com/video/BV1FXgveDEEf/)

---

## 运行的配置（所用的工具和资源）

为了运行本项目，需要确保系统上已安装了所有必要的依赖项，包括Python环境、PyTorch库和OpenCV库。此外，使用具有足够显存的GPU以支持模型的训练过程，如NVIDIA RTX 3050。对于CPU，建议使用性能良好的多核处理器，例如AMD Ryzen 5 5600H系列，以确保足够的计算能力进行图像处理和网络预测。内存至少应为16GB，以便于处理大型图像数据集。

 

使用的包：pytorch, opencv

Python 版本： 3.6

机器配置：GPU为rtx3050 4G，CPU为AMD Ryzen 5 5600H，内存16G

---

## 代码使用方法

使用extend.py文件脚本接收LDR图像作为输入，并创建HDR图像输出

```bash
python expand.py  ldr_input.jpg
```

也可以进行批量处理:

```bash
python expand.py  *.jpg
```

也可以将整个文件包的图片作为输入:

```bash
python expand.py  path/to/ldr_dir
```

把输出的HDR图片存放进文件包:

```bash
python expand.py  *.jpg --out results/
```

可以调整图片大小和更改文件名:
```bash
python expand.py test.jpg --resize True --height 960 --width 540 --tag my-tag
```

可以强制设置cpu使用率：

 `--use_gpu False`.


 生成的HDR图像可以使用OpenCV进行色调映射.

```bash
python expand.py  test.jpg --tone_map reinhard
```

---

## 视频转换

使用 `--video True` (along with a tone mapper `--tone_map`) 可以测试视频转换。

由于我们的能力有限，运行速度较慢且对内存的需求较大，因此应该仅在短时间（3s以内）的低分辨率视频剪辑上进行测。

在以下示例中，左侧是LDR视频输入，右侧是（Reinhard）色调映射的预测结果。
<center>
<img src="assets/Brzansko-Moraviste-Pejzazi_20170226_4457.gif">
</center>
---

## 训练模型

训练:

```bash
python train.py :
```

---

## 绪论

高动态范围成像（High Dynamic Range Imaging，HDRI）技术可以记录和显示场景中更广泛的亮度范围，使图像中的亮部不过曝、暗部不欠曝，从而提供更丰富的视觉细节。传统的单帧图像由于相机传感器的动态范围有限，往往不能同时捕捉到明亮和阴暗区域的细节。因此，多帧图像合成技术应运而生，即通过多张不同曝光的图像合成一张HDR图像。在动态场景中，多帧图像合成存在运动对齐问题，导致重影现象。为了解决这一问题，本文实现了基于深度学习的动态场景HDR成像技术，该技术可以在动态场景中生成高质量的HDR图像。

#### 实验背景与动机

高动态范围成像（HDRI）是一种图像技术，可以捕捉和显示超出传统图像技术的亮度范围。传统的图像技术在处理高对比度场景时往往会导致亮部过曝和暗部欠曝的问题。HDR技术通过组合不同曝光时间的多张图像，可以获得更丰富的细节。然而，在动态场景中，物体的移动会导致图像对齐困难，从而影响HDR图像的质量。我们实现的论文《Deep High Dynamic Range Imaging of Dynamic Scenes》提出了一种基于深度学习的方法，能够在动态场景中生成高质量的HDR图像。

#### 项目目标

我们在实现论文中的方法，并通过实验验证其在不同数据集上的性能。同时，探索并实现一些创新性的改进，提高生成HDR图像的质量。

#### 相关工作

在HDR成像领域，许多人已经提出了不同的方法来解决动态场景中的对齐问题。一些经典的方法包括：

1.多重曝光融合技术

2.图像对齐与去重影算法

3.基于深度学习的HDR生成方法

这些方法各有优缺点，但在处理动态场景时仍存在问题。我们选择了基于深度学习的方法，因为它在处理复杂场景和细节保留方面具有优势。

#### 方法实现

我们的方法基于论文中的ExpandNet网络架构。该网络由三个子网络组成：局部网络、中间网络和全局网络。局部网络负责捕捉局部细节，中间网络提取中间层特征，而全局网络则负责捕捉全局信息。最后，通过融合这三个子网络的输出，生成高质量的HDR图像。

具体实现步骤如下：

1. **预处理输入图像**：对输入的LDR图像进行归一化处理，并根据需要调整图像尺寸。
2. **构建网络模型**：使用PyTorch构建ExpandNet网络，并加载预训练模型权重。
3. **图像处理与预测**：将预处理后的图像输入网络，生成HDR图像。
4. **后处理**：对生成的HDR图像进行色调映射，以便在普通显示设备上显示。

代码实现的主要部分包括：

##### 数据预处理

数据预处理部分包括图像的归一化处理、调整图像尺寸等操作。

```
def process_path(directory, create=False):
    directory = os.path.expanduser(directory)
    directory = os.path.normpath(directory)
    directory = os.path.abspath(directory)
    if create:
        try:
            os.makedirs(directory)
        except:
            pass
    return directory

def map_range(x, low=0, high=1):
    return np.interp(x, [x.min(), x.max()], [low, high]).astype(x.dtype)

def cv2torch(np_img):
    rgb = np_img[:, :, (2, 1, 0)]
    return torch.from_numpy(rgb.swapaxes(1, 2).swapaxes(0, 1))

def torch2cv(t_img):
    return t_img.numpy().swapaxes(0, 2).swapaxes(0, 1)[:, :, (2, 1, 0)]

```

上述函数包括路径处理函数`process_path`，用于将输入路径标准化并创建目录；`map_range`用于将图像数据映射到指定范围；`cv2torch`和`torch2cv`函数则分别用于将OpenCV图像转换为PyTorch张量以及将PyTorch张量转换为OpenCV图像。
在数据预处理阶段，我们需要将图像的像素值映射到指定的范围内使用线性插值公式：
$$
y= 
x 
max
​
 −x 
min
​
 
(x−x 
min
​
 )⋅(b−a)
​
 +a
$$


##### 网络模型

ExpandNet网络模型由三个子网络组成：局部网络（local_net）、中间网络（mid_net）和全局网络（glob_net）。局部网络负责捕捉局部细节，中间网络提取中间层特征，而全局网络则负责捕捉全局信息。
$$
y 
i,j,k
​
 =∑ 
m=0
M−1
​
 ∑ 
n=0
N−1
​
 ∑ 
c=0
C−1
​
 x 
i+m,j+n,c
​
 ⋅w 
m,n,c,k
​
 +b 
k
​
$$

$$
I(x,y)=(1−a)(1−b)I(⌊x⌋,⌊y⌋)+a(1−b)I(⌊x⌋+1,⌊y⌋)+(1−a)bI(⌊x⌋,⌊y⌋+1)+abI(⌊x⌋+1,⌊y⌋+1)
$$



```
import torch
from torch import nn
from torch.nn import functional as F

class ExpandNet(nn.Module):
    def __init__(self):
        super(ExpandNet, self).__init__()

        def layer(nIn, nOut, k, s, p, d=1):
            return nn.Sequential(
                nn.Conv2d(nIn, nOut, k, s, p, d), nn.SELU(inplace=True)
            )

        self.nf = 64
        self.local_net = nn.Sequential(
            layer(3, 64, 3, 1, 1), layer(64, 128, 3, 1, 1)
        )

        self.mid_net = nn.Sequential(
            layer(3, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            layer(64, 64, 3, 1, 2, 2),
            nn.Conv2d(64, 64, 3, 1, 2, 2),
        )

        self.glob_net = nn.Sequential(
            layer(3, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            layer(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 4, 1, 0),
        )

        self.end_net = nn.Sequential(
            layer(256, 64, 1, 1, 0), nn.Conv2d(64, 3, 1, 1, 0), nn.Sigmoid()
        )

    def forward(self, x):
        local = self.local_net(x)
        mid = self.mid_net(x)
        resized = F.interpolate(
            x, (256, 256), mode='bilinear', align_corners=False
        )
        b, c, h, w = local.shape
        glob = self.glob_net(resized).expand(b, 64, h, w)
        fuse = torch.cat((local, mid, glob), -3)
        return self.end_net(fuse)

```



##### 图像处理与预测

图像处理与预测部分包括图像的预处理、网络预测以及后处理等步骤。

```
def preprocess(x, opt):
    x = x.astype('float32')
    if opt.resize:
        x = resize(x, size=(opt.width, opt.height))
    x = map_range(x)
    return x

def create_images(opt):
    net = load_pretrained(opt)
    for ldr_file in opt.ldr:
        loaded = cv2.imread(
            ldr_file, flags=cv2.IMREAD_ANYDEPTH + cv2.IMREAD_COLOR
        )
        if loaded is None:
            print('Could not load {0}'.format(ldr_file))
            continue
        ldr_input = preprocess(loaded, opt)
        if opt.resize:
            out_name = create_name(
                ldr_file, 'resized', 'jpg', opt.out, opt.tag
            )
            cv2.imwrite(out_name, (ldr_input * 255).astype(int))

        t_input = cv2torch(ldr_input)
        if opt.use_gpu:
            net.cuda()
            t_input = t_input.cuda()
        prediction = map_range(
            torch2cv(net.predict(t_input, opt.patch_size).cpu()), 0, 1
        )

        extension = 'exr' if opt.use_exr else 'hdr'
        out_name = create_name(
            ldr_file, 'prediction', extension, opt.out, opt.tag
        )
        print(f'Writing {out_name}')
        cv2.imwrite(out_name, prediction)
        if opt.tone_map is not None:
            tmo_img = tone_map(
                prediction, opt.tone_map, **create_tmo_param_from_args(opt)
            )
            out_name = create_name(
                ldr_file,
                'prediction_{0}'.format(opt.tone_map),
                'jpg',
                opt.out,
                opt.tag,
            )
            cv2.imwrite(out_name, (tmo_img * 255).astype(int))

```

上述代码首先加载预训练的网络模型，然后读取并预处理输入的LDR图像，最后通过网络模型进行预测并保存生成的HDR图像。



除了基本的图像转换功能外，extend.py脚本还提供了多种选项来满足不同的用户需求。例如，用户可以选择是否对最终的HDR图像应用色调映射，以便在不支持HDR显示的设备上查看。此外，脚本允许用户指定输出文件的名称和路径，以及是否将结果保存为流行的EXR格式，这是一种常用于存储HDR图像的文件类型。

高动态范围成像（HDRI）技术的关键在于它能够揭示暗部和亮部的细节，这对于传统的低动态范围（LDR）成像是一个挑战。在电影、摄影和游戏产业中，HDR可以提供更加逼真和具有沉浸感的视觉体验。然而，现有的多帧HDR合成技术在处理包含移动物体的场景时往往效果不佳，因此研究者们开始探索利用深度学习技术来解决这一问题。

我们详细介绍了ExpandNet网络的设计原理，包括各层的具体配置和作用。局部网络使用了较小的感受野来捕捉细节信息，而全局网络则采用了较大的感受野来获取整体结构信息。中间网络则在这两者之间提供了一个过渡。这种设计使得网络能够有效地从不同尺度捕捉信息，并将其融合以产生高质量的HDR图像。

数据预处理步骤至关重要，因为它影响了网络训练的稳定性和效果。我们介绍了如何将原始的LDR图像转换为网络所需的格式，包括归一化、调整尺寸和颜色空间转换。我们还讨论了数据增强的重要性，如随机旋转和翻转，以及如何使用这些技术来增加训练数据的多样性。

进一步详细描述了ExpandNet的网络结构，包括每一层的类型、激活函数的选择以及层与层之间的连接方式。我们还解释了网络如何通过端到端的方式直接从LDR输入预测HDR图像，并讨论了这种方法相对于传统的多步骤方法的优势。

在这一部分，我们详细说明了模型在进行前向传播时的每个步骤，包括如何将LDR图像传递通过网络的不同子模块，并最终生成HDR图像。我们还讨论了如何在后处理阶段应用色调映射，以及如何选择和应用不同的色调映射算法以适应不同的显示设备和用户需求

我们提供了详细的定量和定性实验结果，包括在不同数据集上的比较指标，如峰值信噪比（PSNR）和结构相似性指数（SSIM）。我们还展示了多个视觉示例，对比了我们的方法和现有技术在处理各种动态场景时的HDR图像质量。

## 项目创新

## 1. 基于深度学习的动态场景HDR生成

传统的HDR生成技术在处理动态场景时，往往面临着物体运动导致的图像对齐问题，从而引发重影现象。我们实现的基于深度学习的方法，通过使用ExpandNet网络架构，能够有效地处理动态场景中的运动对齐问题，生成高质量的HDR图像。这在一定程度上克服了传统方法的局限性，使得HDR成像技术在动态场景中同样表现出色。

#### 2. 多尺度特征融合

ExpandNet网络模型包括三个子网络：局部网络、中间网络和全局网络。局部网络负责捕捉图像的细节信息，中间网络提取中间层特征，而全局网络则负责捕捉图像的全局信息。通过融合这三个子网络的输出，模型能够同时捕捉到图像的细节和全局信息，显著提升了HDR图像的质量。这种多尺度特征融合的方式在提高图像细节保留和全局一致性方面具有重要的创新性。

#### 3. 端到端的训练和预测

与传统的多步骤HDR生成方法不同，我们的方法采用端到端的训练和预测方式。模型直接从输入的LDR图像生成HDR图像，不需要中间的手动对齐和图像融合过程。这种端到端的方法不仅简化了处理流程，还提高了处理效率，并且通过深度学习的端到端训练，可以充分利用数据中的隐含信息，提升最终结果的质量。

#### 4. 数据增强与正则化技术

在模型训练过程中，我们引入了多种数据增强技术，包括随机旋转、翻转和颜色扰动等。这些数据增强技术不仅增加了训练数据的多样性，还提高了模型的泛化能力。此外，我们还使用了正则化技术来防止模型过拟合，使得模型在不同的数据集上都能保持良好的性能表现。这些技术手段在深度学习HDR成像领域中起到了关键作用。

#### 5. 自适应色调映射

为了在普通显示设备上显示生成的HDR图像，我们实现了多种色调映射算法，包括Reinhard、Mantiuk、Drago和Durand等。用户可以根据具体需求选择不同的色调映射算法，以适应不同的显示设备和视觉效果需求。这种自适应色调映射的实现，使得HDR图像能够在各种设备上都能呈现出最佳效果。

#### 6. 处理大规模数据的高效实现

为了处理大规模的图像数据，我们在实现过程中采用了图像块处理的方式，以限制内存使用。同时，通过利用GPU加速计算，显著提高了模型的处理速度。我们的实现不仅能够处理高分辨率的图像，还能够在保持高质量的同时，减少计算资源的消耗。这对于实际应用中的大规模图像处理具有重要意义

#### 实验结果：

我们在多个公开数据集上对模型进行了测试，结果如下：

1. **静态场景**：在静态场景中，生成的HDR图像质量较高，细节保留良好。
2. **动态场景**：在动态场景中，模型能够较好地处理运动物体的重影问题，生成的图像没有明显的伪影和失真。

以下是部分实验结果对比图：
<img src="ldr_input.jpg">

<img src="out_put.png">



#### 实验总结：

通过实验，我们验证了基于深度学习的HDR成像方法在低曝光场景中的有效性。模型能够在保持高细节和高动态范围的同时，处理图片过低或过高曝光度的问题。但是模型在处理非常快速的运动物体时，仍存在不足。· 训练过程中，我们采用了特定的损失函数来量化模型的预测与真实HDR图像之间的差异。训练的目标是最小化这一损失，从而优化网络参数。随着训练的进行，我们期望模型将学习到如何从LDR输入中恢复出高质量的HDR图像。为了防止过拟合，我们还引入了正则化技术和数据增强策略。

#### 心得：

在这个项目中，我们通过实现现和改进现有的深度学习模型，深入理解了HDR成像技术的原理和实现过程。在实施过程中，我们面临多个挑战，包括数据集的处理、模型的训练以及实际应用中的性能优化。每一步都需要细致的调试和实验，这不仅提升了我们的技术能力，也让我们对科研工作有了更深刻的认识。学会了如何系统地实现一篇学术论文中的方法。这不仅包括对算法的理解和实现，还涉及到对实验环境的搭建和数据处理的细节。通过这个过程，我们意识到，一个好的算法不仅仅是理论上的创新，更需要在实践中证明其有效性。也体会到了团队合作的重要性。在项目中，吴烨晗主要负责代码的实现和文档的撰写，唐凯则负责数据集的处理和实验设计。通过分工合作，我们能够更高效地推进项目，每个人都发挥了自己的特长。同时，合作过程中，我们也学会了如何沟通和协调，解决了多个技术难题。

另外，通过这个项目，我们深刻理解了数据的重要性。在深度学习项目中，数据的质量和多样性直接影响到模型的性能。为了提升模型的泛化能力，我们进行了大量的数据增强和预处理工作，也会影响最终结果。

## 个人贡献申明

在本项目《Deep High Dynamic Range Imaging of Dynamic Scenes》的实施过程中，两位成员的详细贡献。

#### 吴烨晗

**代码实现**：负责实现ExpandNet网络模型，包括局部网络、中间网络和全局网络的设计与实现。详细定义了网络的每一层、激活函数选择、层与层之间的连接方式，并确保网络结构符合论文要求。编写了数据预处理的代码，负责将输入的LDR图像进行归一化处理、调整图像尺寸，并转换为网络所需的格式。同时，增加了数据增强的技术手段，如随机旋转和翻转，以提升模型的泛化能力。实现了图像的前向传播和预测函数，确保输入的LDR图像能够顺利通过网络，生成高质量的HDR图像。负责编写后处理代码，应用色调映射，将HDR图像转换为LDR格式以便显示。在实现过程中，不断优化代码结构，提高运行效率。进行详细的代码调试，解决实现过程中遇到的各种技术问题，确保模型能够在不同数据集上正常运行。

**文档撰写**：撰写了详细的项目报告，包含实验背景、项目目标、相关工作、方法实现、实验结果、总结与讨论等部分。详细描述了每个步骤的实现过程和技术细节。编写了项目的README文件，提供了项目简介、使用方法、依赖项说明等内容，帮助用户快速了解项目并进行操作。编写了项目的技术文档，包括代码注释、函数说明、网络结构图等，方便其他开发者理解代码实现和网络模型的设计。

#### 唐凯

**代码实现**：在ExpandNet网络模型的实现过程中，协助完成部分子网络的代码编写和调试工作。尤其是在中间网络和全局网络的实现中，提供了重要的技术支持。负责处理项目所需的各类数据集，包括数据集的收集、整理、预处理等工作。确保所有数据集格式统一，能够直接用于网络模型的训练和测试。设计并实施了一系列实验，以验证模型在不同数据集上的性能。包括静态场景和动态场景的测试，确保模型能够在各种复杂场景中生成高质量的HDR图像。

**实验设计**：制定了详细的实验方案，包括实验的具体步骤、所需数据集、评估指标等。确保实验具有科学性和可重复性。负责实验数据的收集和分析，使用定量和定性的方法对实验结果进行评估。包括计算峰值信噪比（PSNR）、结构相似性指数（SSIM）等指标，详细分析模型的性能表现。参与实验结果部分的报告撰写，详细描述实验的实施过程和结果分析。提供实验数据，评估展示模型的性能。

## 引用参考 References

[Deep High Dynamic Range Imaging of Dynamic Scenes](https://people.engr.tamu.edu/nimak/Data/SIGGRAPH17_HDR_LoRes.pdf)

[数据集来源]([MIT-Adobe FiveK dataset](https://data.csail.mit.edu/graphics/fivek/))

邮箱：1575354919@qq.com 
