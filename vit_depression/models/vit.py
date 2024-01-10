import copy
import math

import numpy as np
# from keras.activations import swish
from torch.nn import Dropout
from torch.nn.modules.utils import _pair
import torch
from torch import nn
from torchvision.transforms import transforms


class PatchEmbeddings(nn.Module):
    """
      Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, in_channels=3):
        super(PatchEmbeddings, self).__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        img_size = _pair(img_size) # 生成(img_size x img_size)


        patch_size = _pair(16)
        # 图像块的个数
        num_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=768,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        # 或
        # self.patch_embeddings = in_channels * patch_size[0] * patch_size[1]

        # 位置编码
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, 768))
        # 分类token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))

        self.dropout = Dropout(0.1)

    def forward(self, x): # x:(8,3,224,224)
        B = x.shape[0] # B:8
        cls_tokens = self.cls_token.expand(B, -1, -1) # cls_token分类特征，代码后解释
        #cls_token:(1,1,768) -> cls_tokens:(8,1,768)

        x = self.patch_embeddings(x) # x:(8,768,14,14)
        # x = x.flatten(2)  # x:(8,768,196)
        x =  torch.flatten(x,start_dim=2, end_dim=3) # x:(8,768,196)
        x = x.transpose(-1, -2) # x:(8,196,768)

        x = torch.cat((cls_tokens, x), dim=1) # x:(8,197,768)
        # 将编码向量中加入位置编码
        embeddings = x + self.position_embeddings # position_embeddings:(1,197,768)
        embeddings = self.dropout(embeddings)
        return embeddings



class Multi_Head_Attention(nn.Module):
    def __init__(self):
        super(Multi_Head_Attention, self).__init__()
         # 每个token分配12个头
        self.att_heads_num = 12
        # 经过Embedding层后，输出[8,197,768],表示每个token有768维度，768维度分为12个头处理
        self.att_head_size = int(768 / self.att_heads_num)  # 64
        self.all_head_size = self.att_heads_num * self.att_head_size #768

        self.query = nn.Linear(768, self.all_head_size)
        self.key = nn.Linear(768, self.all_head_size)
        self.value = nn.Linear(768, self.all_head_size)

        self.out = nn.Linear(768, 768)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x): # (8,197,768)
        new_x_shape = \
            x.size()[:-1] + (self.att_heads_num, self.att_head_size)
        x = x.view(new_x_shape) # (8,197,12,64)
        return x.permute(0, 2, 1, 3)  # (8,12,197,64)

    def forward(self, hidden_states): # (8,197,768)
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer) # (8,12,197,64)
        key_layer = self.transpose_for_scores(mixed_key_layer) # (8,12,197,64)
        value_layer = self.transpose_for_scores(mixed_value_layer) # (8,12,197,64)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.att_head_size)
        attention_probs = self.softmax(attention_scores)

        weights =  None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # (8,197,12,64)
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)

        return attention_output, weights


ACT2FN = {"gelu": torch.nn.functional.gelu,
          "relu": torch.nn.functional.relu,
         }


class Mlp(nn.Module):
    def __init__(self):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self):
        super(Block, self).__init__()
        self.hidden_size =768

        self.attention_norm = nn.LayerNorm(768, eps=1e-6)
        self.multi_head_attn = Multi_Head_Attention()

        self.ffn_norm = nn.LayerNorm(768, eps=1e-6)
        self.ffn = Mlp()

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.multi_head_attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(768, eps=1e-6)
        # 堆叠Encoder Block基础模块
        for _ in range(12):
            layer = Block()
            self.layer.append(copy.deepcopy(layer))

    def forward(self, embedding_output):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(embedding_output)

        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Transformer(nn.Module):
    def __init__(self, img_size):
        super(Transformer, self).__init__()
        self.patchEmbeddings = PatchEmbeddings(img_size=img_size)
        self.encoder = Encoder()

    def forward(self, input_ids):
        embedding_output = self.patchEmbeddings(input_ids)
        encoded, attn_weights = self.encoder(embedding_output)
        return encoded, attn_weights
def init_att_layer(layer):
    """
    Initilise the weights and bias of the attention layer to 1 and 0
    respectively. This is because the first iteration through the attention
    mechanism should weight each time step equally.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """
    layer.weight.data.fill_(1.)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    Initialize a Batchnorm layer.

    Input
        bn: torch.Tensor - The batch normalisation layer
    """

    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

def init_layer(layer):
    """Initialize a Linear or Convolutional layer.
    Ref: He, Kaiming, et al. "Delving deep into rectifiers: Surpassing
    human-level performance on imagenet classification." Proceedings of the
    IEEE international conference on computer vision. 2015.

    Input
        layer: torch.Tensor - The current layer of the neural network
    """

    if layer.weight.ndimension() == 4:
        (n_out, n_in, height, width) = layer.weight.size()
        n = n_in * height * width
    elif layer.weight.ndimension() == 3:
        (n_out, n_in, height) = layer.weight.size()
        n = n_in * height
    elif layer.weight.ndimension() == 2:
        (n_out, n) = layer.weight.size()

    std = math.sqrt(2. / n)
    scale = std * math.sqrt(3.)
    layer.weight.data.uniform_(-scale, scale)

    if layer.bias is not None:
        layer.bias.data.fill_(0.)


class FullyConnected(nn.Module):
    """
    Creates an instance of a fully-connected layer. This includes the
    hidden layers but also the type of normalisation "batch" or
    "weight", the activation function, and initialises the weights.
    """
    def __init__(self, in_channels, out_channels, activation, normalisation,
                 att=None):
        super(FullyConnected, self).__init__()
        self.att = att
        self.norm = normalisation
        self.fc = nn.Linear(in_features=in_channels,
                            out_features=out_channels)
        if activation == 'sigmoid':
            self.act = nn.Sigmoid()
            self.norm = None
        elif activation == 'softmax':
            self.act = nn.Softmax(dim=-1)
            self.norm = None
        elif activation == 'global':
            self.act = None
            self.norm = None
        else:
            self.act = nn.ReLU()
            if self.norm == 'bn':
                self.bnf = nn.BatchNorm1d(out_channels)
            elif self.norm == 'wn':
                self.wnf = nn.utils.weight_norm(self.fc, name='weight')

        self.init_weights()

    def init_weights(self):
        """
        Initialises the weights of the current layer
        """
        if self.att:
            init_att_layer(self.fc)
        else:
            init_layer(self.fc)
        if self.norm == 'bn':
            init_bn(self.bnf)

    def forward(self, input):
        """
        Passes the input through the fully-connected layer

        Input
            input: torch.Tensor - The current input at this stage of the network
        """
        x = input
        if self.norm is not None:
            if self.norm == 'bn':
                x = self.act(self.bnf(self.fc(x)))
            else:
                x = self.act(self.wnf(x))
        else:
            if self.att:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)
            else:
                if self.act:
                    x = self.act(self.fc(x))
                else:
                    x = self.fc(x)

        return x


#class VisionTransformer(nn.Module):
class vit(nn.Module):
    def __init__(self, img_size=(1800,72), num_classes=4):
        # super(VisionTransformer, self).__init__()
        super(ConvLSTM_Visual, self).__init__()
        self.num_classes = num_classes
        self.transformer = Transformer(img_size)
        self.head = nn.Linear(768, num_classes)
        self.lstm = nn.LSTM(input_size=768, hidden_size=256, num_layers=2, batch_first=True)
        self.fc = FullyConnected(in_channels=768,
                                 out_channels=256,
                                 activation= 'relu',
                                 normalisation='bn')
    def forward(self, x): # x:(8,3,224,224)
        batch, C, F, T = x.shape
        x, attn_weights = self.transformer(x) # (8,197,768)
        # 提取分类特征Class Token，即第0个token的输出,对所有其他token上的信息做汇聚（全局特征聚合）。
        logits = self.head(x[:, 0])
        x = self.fc(x[:, -1, :].reshape(batch, -1))
        return x

transform = transforms.Compose([
    transforms.Resize(224), #统一图片大小
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225]) #标准化
])




visual_net = vit()
