import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler,Dataset
import copy
import random
import os
import math

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


# mlp dataset
def data_process(file,mode):
    data=pd.read_csv(file)
    data_trm=data['transformers'].tolist()
    data_week=data['is_weekend'].tolist()
    data_mon=data['month'].tolist()
    data_time=data['time'].tolist()
    data_l1=data['L1'].tolist()
    data_l2=data['L2'].tolist()
    data_l3 = data['L3'].tolist()
    data_l4 = data['L4'].tolist()
    data_l5 = data['L5'].tolist()
    data_l6 = data['L6'].tolist()
    data_l6=[list(map(float,[a,b,c,d,e,f])) for a,b,c,d,e,f in zip(data_l1,data_l2,data_l3,data_l4,data_l5,data_l6)]
    data_trm=[int(t_i[-1]) for t_i in data_trm]
    data_week=[int(w_i)+1 for w_i in data_week]
    data_mon=[int(m_i)+1 for m_i in data_mon]
    data_date=data['date_id'].tolist()
    data_date=[int(d_i[1:]) for d_i in data_date]
    data_time=[int(t_i.split(":")[0])*2+int(t_i.split(":")[1])//30+1 for t_i in data_time]
    if mode=="train" or mode=="valid":
        # 获取类别数目：时间、变压器
        trms=len(data["transformers"].unique())
        date=1001
        mon=13
        week=3
        time=49
        embed_num=[trms,date,mon,week,time]
        data_y = data['y'].tolist()
        train_data = []
        for i in range(len(data_y)):
            temp_data=[data_trm[i],data_date[i],data_mon[i],data_week[i],data_time[i]]+data_l6[i]+[data_y[i]]
            train_data.append(temp_data)
        return embed_num, train_data
    else:
        predict_data = []
        for i in range(len(data_trm)):
            temp_data = [data_trm[i], data_date[i], data_mon[i], data_week[i], data_time[i]] + data_l6[i]
            predict_data.append(temp_data)
        return predict_data

# mlp dataset
class Train_Dataset(Dataset):
    def __init__(self, args,data,mode):
        self.args = args
        self.data=data
        self.mode=mode

    def __getitem__(self, index):
        data = self.data[index]
        if self.mode=="train":
            cur_train_tensor = (
            torch.tensor(data[0],dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[1], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[2], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[3], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[4], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[5:-1],dtype=torch.float).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[-1],dtype=torch.float).to("cuda:%s"%self.args.gpu_id),
            )
        elif self.mode=="valid":
            cur_train_tensor = (
                torch.tensor(data[0], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[1], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[2], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[3], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[4], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[5:-1], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[-1], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            )
        else:
            cur_train_tensor = (
                torch.tensor(data[0], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[1], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[2], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[3], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[4], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[5:], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            )

        return cur_train_tensor

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.data)
# 全局标准化
def min_max(a):
    ma=max(a)
    mi=min(a)
    res=[]
    for i in a:
        scale_i=-1+2*(i-mi)/(ma-mi)
        res.append(scale_i)
    return res
# 局部标准化
def min_max_(a):
    res=[]
    for i in range(len(a)//48):
        pre_id, rear_id = i * 48, (i + 1) * 48
        temp_mi=min(a[pre_id:rear_id])
        temp_ma=max(a[pre_id:rear_id])
        for j in a[pre_id:rear_id]:
            scale_j=-1+2*(j-temp_mi)/(temp_ma-temp_mi)
            res.append(scale_j)
    return res

# 局部加入均值和方差
def mean_std_(a):
    res=[]
    for i in range(len(a)//48):
        pre_id, rear_id = i * 48, (i + 1) * 48
        temp_list=np.array(a[pre_id:rear_id])
        temp_mean=np.mean(temp_list).tolist()
        temp_std=np.var(temp_list).tolist()
        # print(temp_mean,temp_std)
        for j in a[pre_id:rear_id]:
            scale_j=j+temp_mean+temp_std
            res.append(scale_j)
    return res

# 反标准化
def f_min_max(a,mi,delta):
    res=[]
    for i in a:
        fscale_i=i*delta+mi
        res.append(fscale_i)
    return res


def Mask(seq,mask_len,mask_idx):
    """
    :param seq: 原始序列
    :param mask_ratio: 掩盖比例
    :param n: 原始序列长度
    :return: mask之后的序列
    """
    seq = copy.deepcopy(seq)
    seq=np.array(seq)
    mask_id=0
    if mask_len<1:
        return seq.tolist()
    seq[mask_idx]=mask_id
    return seq.tolist()

# 随机删除一部分item
def Crop_single(seq,crop_len,crop_idx):
    """
    :param seq: 原始序列
    :param crop_ratio: 剪切比例
    :param n: 原始序列长度
    :return: crop之后的序列
    """
    seq = copy.deepcopy(seq)
    seq = np.array(seq)
    if crop_len<1:
        return seq.tolist()
    seq=np.delete(seq,crop_idx)
    seq=seq.tolist()
    seq=[0]*crop_len+seq
    return seq


# 随机打乱一部分item顺序
def Reorder(seq,reorder_len,start):
    """
    :param seq: 原始序列
    :param reorder_ratio: 打乱比例
    :param n: 原始序列长度
    :return: reorder之后的序列
    """
    seq = copy.deepcopy(seq)
    seq = np.array(seq)
    if reorder_len<1:
        return seq.tolist()
    sub_seq=seq[start:start+reorder_len]
    random.shuffle(sub_seq)
    seq[start:start+reorder_len]=sub_seq
    return seq.tolist()
# if __name__ == '__main__':
#     a=np.array([i for i in range(20)])
#     mask_ratio=0.5
#     n=20
#     # print(type(mask_op(a,mask_ratio,n)))


def data_aug(seqs,types,ratios):
    """
    :param seqs: [11x800]
    :param types: random smple two from ["mask","crop","reorder"]
    :param ratios: [mask_ratio,crop_ratio,reorder_ratio]
    :return: [aug_seq0,aug_seq1]
    """
    # 操作比例
    mask_len =  int(48 * ratios[0])
    mask_idx = random.sample(range(48), mask_len)
    crop_len =  int(48 * ratios[1])
    crop_idx = random.sample(range(48), crop_len)
    reorder_len = int(48*ratios[2])
    reorder_start=random.randint(0,48-reorder_len)
    seqs=copy.deepcopy(seqs)
    aug_0=[]
    aug_1=[]
    for s in seqs:
        mask_res=Mask(s,mask_len,mask_idx)
        crop_res=Crop_single(s,crop_len,crop_idx)
        reorder_res=Reorder(s,reorder_len,reorder_start)
        aug_res={"mask":mask_res,"crop":crop_res,"reorder":reorder_res}
        aug_0.append(aug_res[types[0]])
        aug_1.append(aug_res[types[1]])
    return aug_0,aug_1


# transformer dataset
def t_data_process(file,mode):
    data=pd.read_csv(file)
    data_trm=data['transformers'].tolist()
    data_date = data['date_id'].tolist()
    data_week=data['is_weekend'].tolist()
    data_mon=data['month'].tolist()
    data_time=data['time'].tolist()
    data_l1 = data['L1'].tolist()
    data_l2 = data['L2'].tolist()
    data_l3 = data['L3'].tolist()
    data_l4 = data['L4'].tolist()
    data_l5 = data['L5'].tolist()
    data_l6 = data['L6'].tolist()

    # 转换数据类型
    data_trm=[int(t_i[-1]) for t_i in data_trm]
    data_date=[int(d_i[1:]) for d_i in data_date]
    data_week=[int(w_i)+1 for w_i in data_week]
    data_mon=[int(m_i) for m_i in data_mon]
    data_time=[int(t_i.split(":")[0])*2+int(t_i.split(":")[1])//30+1 for t_i in data_time]


    data_l1 = [float(d) for d in data_l1]
    data_l2 = [float(d) for d in data_l2]
    data_l3 = [float(d) for d in data_l3]
    data_l4 = [float(d) for d in data_l4]
    data_l5 = [float(d) for d in data_l5]
    data_l6 = [float(d) for d in data_l6]

    # 对于数值型特征进行归一化处理，按照序列长度来
    # print(max(data_trm),max(data_date),max(data_week),max(data_mon),max(data_time))
    data_l1 = mean_std_(data_l1)
    data_l2 = mean_std_(data_l2)
    data_l3 = mean_std_(data_l3)
    data_l4 = mean_std_(data_l4)
    data_l5 = mean_std_(data_l5)
    data_l6 = mean_std_(data_l6)



    # 数据合并
    if mode=="train" or mode=="valid":
        # 获取类别数目：时间、变压器
        trms=3
        date=1001
        mon=13
        week=3
        time=49
        embed_num=[trms,date,mon,week,time]
        data_y = data['y'].tolist()
        data_y=[float(d) for d in data_y]
        # mi,ma=min(data_y),max(data_y)
        # data_y = min_max(data_y)
        train_data = []
        for i in range(len(data_y)//48):
            pre_id,rear_id=i*48,(i+1)*48
            temp_data=[data_trm[pre_id:rear_id],data_date[pre_id:rear_id],data_mon[pre_id:rear_id],data_week[pre_id:rear_id],data_time[pre_id:rear_id],data_l1[pre_id:rear_id],data_l2[pre_id:rear_id],data_l3[pre_id:rear_id],data_l4[pre_id:rear_id],data_l5[pre_id:rear_id],data_l6[pre_id:rear_id],data_y[pre_id:rear_id]]
            train_data.append(temp_data)

        aug_seq_0=[]
        aug_seq_1=[]
        aug_ratios=[0.2,0.2,0.2]
        for seq in train_data:
            augtypes=random.sample(["mask","crop","reorder"],2)
            aug_seq_0_t,aug_seq_1_t=data_aug(seq,augtypes,aug_ratios)
            # print("ori: ", seq)
            # print("aug0: ",aug_seq_0_t)
            # print("aug1: ",aug_seq_1_t)
            aug_seq_0.append(aug_seq_0_t)
            aug_seq_1.append(aug_seq_1_t)

        return embed_num, train_data,aug_seq_0,aug_seq_1
    else:
        predict_data = []
        for i in range(len(data_trm)//48):
            pre_id, rear_id = i * 48, (i + 1) * 48
            temp_data = [data_trm[pre_id:rear_id], data_date[pre_id:rear_id],data_mon[pre_id:rear_id],data_week[pre_id:rear_id], data_time[pre_id:rear_id],data_l1[pre_id:rear_id], data_l2[pre_id:rear_id], data_l3[pre_id:rear_id], data_l4[pre_id:rear_id],data_l5[pre_id:rear_id], data_l6[pre_id:rear_id]]
            predict_data.append(temp_data)

        return predict_data


class NCELoss(nn.Module):
    """
    Eq. (12): L_{NCE}
    """

    def __init__(self, temperature, device):
        super(NCELoss, self).__init__()
        self.device = device
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1).to(self.device)

    # #modified based on impl: https://github.com/ae-foster/pytorch-simclr/blob/dc9ac57a35aec5c7d7d5fe6dc070a975f493c1a5/critic.py#L5
    def forward(self, batch_sample_one, batch_sample_two, intent_ids=None):
        # sim11 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_one.unsqueeze(-3)) / self.temperature
        # sim22 = self.cossim(batch_sample_two.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        # sim12 = self.cossim(batch_sample_one.unsqueeze(-2), batch_sample_two.unsqueeze(-3)) / self.temperature
        sim11 = torch.matmul(batch_sample_one, batch_sample_one.T) / self.temperature
        sim22 = torch.matmul(batch_sample_two, batch_sample_two.T) / self.temperature
        sim12 = torch.matmul(batch_sample_one, batch_sample_two.T) / self.temperature
        d = sim12.shape[-1]
        # avoid contrast against positive intents
        if intent_ids is not None:
            intent_ids = intent_ids.contiguous().view(-1, 1)
            mask_11_22 = torch.eq(intent_ids, intent_ids.T).long().to(self.device)
            sim11[mask_11_22 == 1] = float("-inf")
            sim22[mask_11_22 == 1] = float("-inf")
            eye_metrix = torch.eye(d, dtype=torch.long).to(self.device)
            mask_11_22[eye_metrix == 1] = 0
            sim12[mask_11_22 == 1] = float("-inf")
        else:
            mask = torch.eye(d, dtype=torch.long).to(self.device)
            sim11[mask == 1] = float("-inf")
            sim22[mask == 1] = float("-inf")
            # sim22 = sim22.masked_fill_(mask, -np.inf)
            # sim11[..., range(d), range(d)] = float('-inf')
            # sim22[..., range(d), range(d)] = float('-inf')

        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        nce_loss = self.criterion(logits, labels)
        return nce_loss



# transformer dataset
class S2S_Dataset(Dataset):
    """
    trm_id： [[1,2,1,...,1,2],...]
    time: [[a,b,c,...,a,b],...]
    l1-l6: [[1,2,3,...,6],...]
    """
    def __init__(self,args,data,mode):
        super(S2S_Dataset, self).__init__()
        self.args=args
        # 补充
        # self.max_len=self.args.max_sequence_length
        self.data=data
        self.mode=mode
    def __getitem__(self, index):
        data = self.data[index]
        if self.mode=="train" or self.mode=="valid":
            cur_train_tensor = (
            torch.tensor(data[0],dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[1], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[2], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[3], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[4], dtype=torch.long).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[5],dtype=torch.float).to("cuda:%s"%self.args.gpu_id),
            torch.tensor(data[6], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            torch.tensor(data[7], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            torch.tensor(data[8], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            torch.tensor(data[9], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            torch.tensor(data[10], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            torch.tensor(data[11], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            )
        else:
            cur_train_tensor = (
                torch.tensor(data[0], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[1], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[2], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[3], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[4], dtype=torch.long).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[5], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[6], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[7], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[8], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[9], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
                torch.tensor(data[10], dtype=torch.float).to("cuda:%s" % self.args.gpu_id),
            )

        return cur_train_tensor

    def __len__(self):
        """
        consider n_view of a single sequence as one sample
        """
        return len(self.data)



def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """

    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)  # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        # 修改属性
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads)
            )
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        # 做完self-attention 做一个前馈全连接 LayerNorm 输出
        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        # print("input: ",input_tensor)
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]

        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        # print("hidden: ",hidden_states)
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # 参数设置
    parser.add_argument('--gpu_id',type=str,default='0')
    parser.add_argument('--data_dir',type=str,default='train.csv')
    parser.add_argument('--trm_hidden_size',type=int,default=128)
    parser.add_argument('--date_hidden_size',type=int,default=128)
    parser.add_argument('--mon_hidden_size', type=int, default=128)
    parser.add_argument('--week_hidden_size', type=int, default=128)
    parser.add_argument('--time_hidden_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--l_num', type=int, default=6)
    parser.add_argument('--early_stop', type=int, default=10)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--max-sequence_length", type=int, default=48, help="adam second beta value")

    # 数据加载与处理
    classes_num, data= data_process("train.csv","train")
    args=parser.parse_args()

    # a,data=data_process("train.csv","train")
    # train_data=(data)
    # cur_train_tensor=Train_Dataset(a,train_data,"train")
    # train_sampler=SequentialSampler(cur_train_tensor)
    # train_dataloader=DataLoader(cur_train_tensor,sampler=train_sampler,batch_size=48)
    # for batch_data in train_dataloader:
    #     print("batch: ",len(batch_data),batch_data[5])
    #     break
    class_,data=t_data_process("train.csv","train")
    # print("data: ",data,len(data[0]))
    # print("label",label,len(label[0]))
    train_data=(data)
    cur_train_tensor=S2S_Dataset(args,train_data,"train")
    train_sampler=SequentialSampler(cur_train_tensor)
    train_dataloader=DataLoader(cur_train_tensor,sampler=train_sampler,batch_size=192)
    for batch_data in train_dataloader:
        # print("batch: ",len(batch_data),batch_data[0].shape)
        # classes=batch_data[:5]
        # print("classes4: ",classes[4],nn.Embedding(49,64)(classes[4]))
        # ldata=batch_data[5:-1]
        label=batch_data[-1]
        # print("cless" ,classes)
        # print("ldata: ",ldata)
        # print("label: ",label)
        # break



