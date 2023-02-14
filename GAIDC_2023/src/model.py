import os
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import SequentialSampler,DataLoader,RandomSampler
from utils import data_process,Train_Dataset,set_seed,t_data_process,S2S_Dataset,NCELoss
from torch.nn import MSELoss,L1Loss
import matplotlib.pyplot as plt
from utils import Encoder, LayerNorm


# MLP
class MLP(nn.Module):
    def __init__(self,args):
        super(MLP, self).__init__()
        self.trm_embedding=nn.Embedding(args.trms+1,args.trm_hidden_size,padding_idx=0)
        self.date_embedding=nn.Embedding(args.date+1,args.date_hidden_size,padding_idx=0)
        self.month_embedding=nn.Embedding(args.month+1,args.mon_hidden_size,padding_idx=0)
        self.weekend_embedding=nn.Embedding(args.week+1,args.week_hidden_size,padding_idx=0)
        self.time_embedding=nn.Embedding(args.time+1,args.time_hidden_size,padding_idx=0)
        self.L6_0=nn.Linear(args.l_num,args.hidden_size)
        self.drop=nn.Dropout(args.drop_rate)
        self.L6_1=nn.Linear(args.hidden_size,args.hidden_size)
        # self.act_func=ActFun(args.act_func)
        self.L6_2=nn.Linear(args.hidden_size,args.hidden_size)
        self.res=nn.Linear(args.hidden_size,1)
    def embedding_init(self,classes):
        trm_emb=self.trm_embedding(classes[0])
        date_emb=self.date_embedding(classes[1])
        mon_emb=self.month_embedding(classes[2])
        week_emb=self.weekend_embedding(classes[3])
        time_emb=self.time_embedding(classes[4])
        total_time_emb=date_emb+mon_emb+week_emb+time_emb
        return trm_emb,total_time_emb

    def forward(self,data):
        # [B,classes-l6]
        classes,l6=data
        trm_emb,time_emb=self.embedding_init(classes)
        output=self.L6_0(l6)
        output=self.L6_1(output)
        output=self.L6_2(output)
        output=trm_emb+time_emb+output
        # output=self.drop(output)+output
        output=self.res(output)
        return output

# transformer
class MyTransformer(nn.Module):
    def __init__(self,args):
        super(MyTransformer, self).__init__()
        self.args=args
        self.trm_embedding = nn.Embedding(args.trms , args.trm_hidden_size, padding_idx=0)
        self.date_embedding = nn.Embedding(args.date , args.date_hidden_size, padding_idx=0)
        self.month_embedding = nn.Embedding(args.month, args.mon_hidden_size, padding_idx=0)
        self.weekend_embedding = nn.Embedding(args.week, args.week_hidden_size, padding_idx=0)
        self.time_embedding = nn.Embedding(args.time, args.time_hidden_size, padding_idx=0)

        self.embed_weight=nn.Parameter(torch.ones(1,5),requires_grad=True)
        self.l6_weight=nn.Parameter(torch.ones(1,6),requires_grad=True)

        self.L1 = nn.Linear(1, args.hidden_size)
        self.L2 = nn.Linear(1, args.hidden_size)
        self.L3 = nn.Linear(1, args.hidden_size)
        self.L4 = nn.Linear(1, args.hidden_size)
        self.L5 = nn.Linear(1, args.hidden_size)
        self.L6 = nn.Linear(1, args.hidden_size)

        self.l1  = nn.Linear(args.hidden_size,args.hidden_size)
        self.l2  = nn.Linear(args.hidden_size, args.hidden_size)
        self.res = nn.Linear(args.hidden_size, 1)

        self.item_encoder = Encoder(args)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.drop_rate)

    def embedding_init(self, classes,ldata):
        # classes embedding, BxLxH
        trm_emb = self.trm_embedding(classes[0])
        date_emb = self.date_embedding(classes[1])
        mon_emb = self.month_embedding(classes[2])
        week_emb = self.weekend_embedding(classes[3])
        time_emb = self.time_embedding(classes[4])
        total_time_emb = date_emb + mon_emb + week_emb + time_emb
        seq_class_emb=trm_emb+total_time_emb
        # seq_class_emb=self.embed_weight.data*[trm_emb,date_emb,mon_emb,week_emb,time_emb]


        # l embedding, BxH
        # print(ldata[0].shape,ldata[0])
        l1 = self.L1(ldata[0].unsqueeze(dim=2))
        l2 = self.L2(ldata[1].unsqueeze(dim=2))
        l3 = self.L3(ldata[2].unsqueeze(dim=2))
        l4 = self.L4(ldata[3].unsqueeze(dim=2))
        l5 = self.L5(ldata[4].unsqueeze(dim=2))
        l6 = self.L6(ldata[5].unsqueeze(dim=2))
        total_l_emb = l1 + l2 + l3 + l4 + l5 + l6
        # total_l_emb=self.l6_weight*[l1,l2,l3,l4,l5,l6]

        total_l_emb=self.l1(total_l_emb)
        total_l_emb=self.l2(total_l_emb)

        # lmd=0.9
        seq_emb=seq_class_emb+total_l_emb
        seq_emb=self.LayerNorm(seq_emb)
        seq_emb=self.dropout(seq_emb)
        return seq_emb

    # model same as SASRec
    def forward(self, time_series):
        # [B,L]
        classes,ldata= time_series
        # print("classes: ",classes,len(classes),classes[0].shape)
        # print("ldata: ",ldata,len(ldata),ldata[0].shape)
        seq_emb = self.embedding_init(classes,ldata)

        attention_mask = (classes[4] > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        item_encoded_layers = self.item_encoder(seq_emb, extended_attention_mask, output_all_encoded_layers=True)
        sequence_output = item_encoded_layers[-1]

        sequence_output=self.res(sequence_output).view(sequence_output.shape[0],-1)

        # sequence_output=torch.mean(sequence_output,dim=2,keepdim=False)+sequence_output_1
        # print(sequence_output.shape)

        return sequence_output



def show_args_info(args):
    print(f"--------------------Configure Info:------------")
    for arg in vars(args):
        print(f"{arg:<30} : {getattr(args, arg):>35}")


def save(file_name,model):
    torch.save(model.cpu().state_dict(), file_name)
    model.cuda()

def load(file_name,model):
    model.load_state_dict(torch.load(file_name))
    model.cuda()



def train_mlp(model,args,train_dataloader,valid_dataloader,optimizer):
    # 模型学习与验证
    train_losses = []
    early_stop_loss = 0
    stop_num = 0
    for e_p in range(args.train_epoch):
        train_loss = 0.
        valid_loss=0.
        for train_batch_data in train_dataloader:
            class_data = train_batch_data[:5]
            l6_data = train_batch_data[5:-1][0]
            target_y = train_batch_data[-1]
            output = model((class_data, l6_data)).view(-1)
            train_loss += nn.L1Loss(reduction="sum")(output,target_y)*100

        print("Training at epoch {0}: training loss {1} ".format(e_p, train_loss))

        for valid_batch_data in valid_dataloader:
            class_data = valid_batch_data[:5]
            l6_data = valid_batch_data[5:-1][0]
            target_y = valid_batch_data[-1]
            output = model((class_data, l6_data)).view(-1)
            valid_loss = nn.L1Loss(reduction="mean")(output, target_y) * 100
            valid_loss=valid_loss.detach()
        if stop_num == 0:
            early_stop_loss = valid_loss
            stop_num += 1
        else:
            if early_stop_loss >= valid_loss:
                early_stop_loss = valid_loss
                save("mlp.pt", model)
                stop_num = 1
            else:
                print("Early Stopping {0} at epoch {1}: training loss {2} valid loss {3}".format(stop_num, e_p,
                                                                                                 train_loss,
                                                                                                valid_loss))
                stop_num += 1
        if stop_num == args.early_stop + 1:
            break

        train_losses.append(train_loss.detach().cpu())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

def test_mlp(model, args, predict_dataloader,data_name):
    load("mlp.pt", model)
    # 模型预测
    res = []
    for predict_batch_data in predict_dataloader:
        # print(len(predict_batch_data[5:]))
        class_data = predict_batch_data[:5]
        l6_data = predict_batch_data[5:][0]
        # print(l6_data)
        output = model((class_data, l6_data))
        res.append(output.detach().cpu().numpy().tolist())
    res_predict = []
    for r in res:
        for res_b in r:
            for res_value in res_b:
                res_predict.append(res_value)
    data_predict = pd.read_csv("preliminary_%s_submit.csv"%data_name)
    data_predict["y"] = res_predict
    data_predict.to_csv("%s_result_%s_%d_.csv" % (args.model_name,data_name,args.train_epoch), index=False)


def train_transformer(model,args,train_dataloader,aug0,aug1,valid_dataloader,optimizer):
    # 模型学习与验证
    cl_loss = NCELoss(1.0, "cuda:0")
    train_losses = []
    valid_losses=[]
    early_stop_loss = 0
    stop_num = 0
    for e_p in range(args.train_epoch):
        train_loss = 0.
        aug_loss=0.
        for train_batch_data,aug_batch_data_0,aug_batch_data_1 in zip(train_dataloader,aug0,aug1):
            #-----------------------predict Loss-------------------------------
            class_data = train_batch_data[:5]
            l6_data = train_batch_data[5:-1]
            target_y = train_batch_data[-1]
            output = model((class_data, l6_data))
            train_loss += nn.L1Loss(reduction="mean")(output,target_y)*100
            #----------------------contrastive Loss-------------------------------
            class_data_0=aug_batch_data_0[:5]
            l6_data_0=aug_batch_data_0[5:-1]
            aug_0_output=model((class_data_0,l6_data_0))
            class_data_1 = aug_batch_data_1[:5]
            l6_data_1 = aug_batch_data_1[5:-1]
            aug_1_output = model((class_data_1, l6_data_1))
            aug_loss +=torch.sum(nn.CosineSimilarity()(aug_0_output,aug_1_output))

        joint_loss = 1.0 * train_loss + 0.1 * aug_loss
        print("Training at epoch {0}: training loss {1} aug loss {2} joint loss {3}".format(e_p, train_loss,aug_loss,joint_loss))

        for valid_batch_data in valid_dataloader:
            class_data = valid_batch_data[:5]
            l6_data = valid_batch_data[5:-1]
            target_y = valid_batch_data[-1]
            output = model((class_data, l6_data))
            valid_loss = nn.L1Loss(reduction="mean")(output, target_y) * 100
            valid_loss=valid_loss.detach()
        if stop_num == 0:
            early_stop_loss = valid_loss
            stop_num += 1
        else:
            if early_stop_loss >= valid_loss:
                early_stop_loss = valid_loss
                save("trm.pt", model)
                stop_num = 1
            else:
                print("Early Stopping {0} at epoch {1}: training loss {2} valid loss {3}".format(stop_num, e_p,
                                                                                                 train_loss,
                                                                                                valid_loss))
                stop_num += 1
        if stop_num == args.early_stop + 1:
            break

        train_losses.append(train_loss.detach().cpu())
        optimizer.zero_grad()
        joint_loss.backward()
        optimizer.step()

def test_transformer(model, args, predict_dataloader,data_name):
    load("trm.pt", model)
    # 模型预测
    res = []
    for predict_batch_data in predict_dataloader:
        # print(len(predict_batch_data[5:]))
        class_data = predict_batch_data[:5]
        l6_data = predict_batch_data[5:]
        output = model((class_data, l6_data))
        res.append(output.detach().cpu().numpy().tolist())
    res_predict = []
    for r in res:
        for res_b in r:
            for res_value in res_b:
                res_predict.append(res_value)
    data_predict = pd.read_csv("preliminary_%s_submit.csv"%data_name)
    data_predict["y"] = res_predict
    data_predict.to_csv("%s_result_%s_%d_.csv" % (args.model_name,data_name,args.train_epoch), index=False)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    # 参数设置
    parser.add_argument('--model_name', type=str, default='Trm')
    parser.add_argument('--gpu_id',type=str,default='0')
    parser.add_argument("--log_freq", type=int, default=1, help="per epoch print res")
    parser.add_argument('--data_dir',type=str,default='train.csv')
    parser.add_argument('--trm_hidden_size',type=int,default=64)
    parser.add_argument('--date_hidden_size',type=int,default=64)
    parser.add_argument('--mon_hidden_size', type=int, default=64)
    parser.add_argument('--week_hidden_size', type=int, default=64)
    parser.add_argument('--time_hidden_size', type=int, default=64)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--train_epoch', type=int, default=1000)
    parser.add_argument('--l_num', type=int, default=6)
    parser.add_argument('--early_stop', type=int, default=20)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of adam")
    parser.add_argument("--weight_decay", type=float, default=0.5, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.5, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.5, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=48)


    args = parser.parse_args()

    # 数据加载与处理
    if args.model_name=="MLP":
        classes_num, data= data_process("train.csv","train")
        args.trms,args.date,args.month,args.week,args.time=classes_num
        train_data = (data)
        # 训练集
        cur_train_tensor = Train_Dataset(args, train_data,"train")
        train_sampler = SequentialSampler(cur_train_tensor)
        train_dataloader = DataLoader(cur_train_tensor, sampler=train_sampler, batch_size=args.batch_size)
        show_args_info(args)
        # 需要预测的数据
        predict_data=data_process("preliminary_A.csv","predict")
        predict_data=(predict_data)
        cur_predict_tensor=Train_Dataset(args,predict_data,"predict")
        predict_sampler = SequentialSampler(cur_predict_tensor)
        # print(len(predict_data))
        predict_dataloader = DataLoader(cur_predict_tensor, sampler=predict_sampler, batch_size=len(predict_data))
    else:
        classes_num,data=t_data_process("train.csv","train")
        args.trms,args.date,args.month,args.week,args.time=classes_num
        random.shuffle(data)
        # print("length: ",len(data),len(data[0]))
        # train_data = (data[:int(len(data)*0.75)])
        train_data = (data)
        # print("train length: ",len(train_data))
        # print(len(train_data[0]))
        # 训练集
        cur_train_tensor = S2S_Dataset(args, train_data,"train")
        train_sampler = RandomSampler(cur_train_tensor)
        train_dataloader = DataLoader(cur_train_tensor, sampler=train_sampler, batch_size=args.batch_size)
        show_args_info(args)

        # 验证集
        valid_data=(data[int(0.75*len(data)):])
        # print("valid length: ",len(valid_data))
        cur_valid_tensor=S2S_Dataset(args,valid_data,"valid")
        valid_sampler = SequentialSampler(cur_valid_tensor)
        valid_dataloader=DataLoader(cur_valid_tensor,sampler=valid_sampler,batch_size=len(valid_data))

        # 需要预测的数据
        predict_data=t_data_process("preliminary_B.csv","predict")
        predict_data=(predict_data)
        cur_predict_tensor=S2S_Dataset(args,predict_data,"predict")
        predict_sampler = SequentialSampler(cur_predict_tensor)
        predict_dataloader = DataLoader(cur_predict_tensor, sampler=predict_sampler, batch_size=len(predict_data))



    # 模型初始化
    if args.model_name=="MLP":
        model=MLP(args=args)
    else:
        model=MyTransformer(args=args)

    model.cuda()
    # 使用预训练的模型参数来进行初始化
    # load("mlp.pt", model)

    # 设置环境与随机数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    set_seed(args.seed)
    # 设置优化器以及相关参数
    betas = (args.adam_beta1, args.adam_beta2)
    optimizer=Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
    # train_mlp(model,args,train_dataloader,predict_dataloader)
    # train_transformer(model,args,train_dataloader,valid_dataloader,predict_dataloader,optimizer)
















