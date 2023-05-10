import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import utils as nn_utils

class GRU_Attention(nn.Module):

    def __init__(self, input_size):
        super().__init__()
        self.BiGRU_profile = nn.GRU(input_size=input_size, hidden_size=512, num_layers=4, bidirectional=True)

    def attention_net(self, BiGRU_output, final_state):
        batch_size = len(BiGRU_output)  # 计算每一个批次的大小
        hidden = final_state.view(batch_size, -1,
                                  1)  # hidden : [batch_size, n_hidden * num_directions(=2), n_layer(=1)]
        attn_weights = torch.bmm(BiGRU_output, hidden).squeeze(2)  # attn_weights : [batch_size, n_step] 计算权重
        soft_attn_weights = F.softmax(attn_weights, 1)  # 权重列向量归一化处理
        context = torch.bmm(BiGRU_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(
            2)  # 将权重和BiGRU的输出结果相乘之后加权求和
        return context, soft_attn_weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs, h2 = self.BiGRU_profile(x)
        outputs = nn_utils.rnn.pad_packed_sequence(outputs, batch_first=True)  # 重新填充BiGRU的输出结果
        output = outputs[0]  # 重新填充后的数据
        length = outputs[1].numpy()  # rna序列的原始长度，不算填充的
        out = output[:, -1, :]  # reshape一下为了输入到AM中
        out_Put = output[:, 0:200, :]  # AM的step设置为200
        for number in range(len(length)):
            out[number] = output[number][length[number] - 1]  # 给reshape后的out赋值
        attn_output, attention = self.attention_net(out_Put, out)  # attn——output为注意力机制输出的结果
        out = attn_output.reshape(-1, 1, 32, 32)
        return out

def GRU_word2vec():
    return GRU_Attention(input_size=100)
def GRU_DCN():
    return GRU_Attention(input_size=4)
def GRU_kmer():
    return GRU_Attention(input_size=12)
def GRU_GCN():
    return GRU_Attention(input_size=16)