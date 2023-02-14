from model import *
from utils import *



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
    parser.add_argument("--weight_decay", type=float, default=0.1, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam second beta value")
    parser.add_argument("--num_hidden_layers", type=int, default=2, help="number of layers")
    parser.add_argument("--num_attention_heads", default=8, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", type=float, default=0.1, help="attention dropout p")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="hidden dropout p")
    parser.add_argument("--initializer_range", type=float, default=0.02)
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--do_eval", action="store_true")


    args = parser.parse_args()

    # 数据加载与处理
    if args.model_name=="MLP":
        classes_num, data= data_process("train.csv","train")
        args.trms,args.date,args.month,args.week,args.time=classes_num
        train_data = (data[:int(0.75 * len(data))])
        # 训练集
        cur_train_tensor = Train_Dataset(args, train_data,"train")
        train_sampler = SequentialSampler(cur_train_tensor)
        train_dataloader = DataLoader(cur_train_tensor, sampler=train_sampler, batch_size=args.batch_size)
        show_args_info(args)
        # 验证集
        valid_data = (data[int(0.75 * len(data)):])
        # print("valid length: ",len(valid_data))
        cur_valid_tensor = Train_Dataset(args, valid_data, "valid")
        valid_sampler = SequentialSampler(cur_valid_tensor)
        valid_dataloader = DataLoader(cur_valid_tensor, sampler=valid_sampler, batch_size=len(valid_data))

        # 需要预测的数据
        predict_data=data_process("preliminary_B.csv","predict")
        predict_data=(predict_data)
        cur_predict_tensor=Train_Dataset(args,predict_data,"predict")
        predict_sampler = SequentialSampler(cur_predict_tensor)
        # print(len(predict_data))
        predict_dataloader = DataLoader(cur_predict_tensor, sampler=predict_sampler, batch_size=len(predict_data))
    else:
        classes_num,data,aug_0,aug_1=t_data_process("train.csv","train")
        args.trms,args.date,args.month,args.week,args.time=classes_num

        train_data = (data[:int(0.75*len(data))])
        aug_data_0 = (aug_0)
        aug_data_1 = (aug_1)

        # 训练集
        cur_train_tensor= S2S_Dataset(args, train_data,"train")
        cur_aug_tensor_0= S2S_Dataset(args,aug_data_0,"train")
        cur_aug_tensor_1= S2S_Dataset(args,aug_data_1,"train")

        train_sampler = SequentialSampler(cur_train_tensor)
        aug_sampler_0=SequentialSampler(cur_aug_tensor_0)
        aug_sampler_1= SequentialSampler(cur_aug_tensor_1)

        # 增强视图
        train_dataloader = DataLoader(cur_train_tensor, sampler=train_sampler, batch_size=args.batch_size)
        aug_dataloader0=DataLoader(cur_aug_tensor_0,sampler=aug_sampler_0,batch_size=args.batch_size)
        aug_dataloader1=DataLoader(cur_aug_tensor_1,sampler=aug_sampler_1,batch_size=args.batch_size)
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


    # 设置环境与随机数
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print("Using Cuda:", torch.cuda.is_available())
    set_seed(args.seed)
    # 设置优化器以及相关参数
    betas = (args.adam_beta1, args.adam_beta2)

    # 模型初始化
    if args.model_name=="MLP":
        model=MLP(args=args)
        model.cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
        if args.do_eval:
            load("mlp.pt", model)
            test_mlp(model, args, predict_dataloader, "b")
        else:
            train_mlp(model,args,train_dataloader,valid_dataloader,optimizer)

    else:
        model=MyTransformer(args=args)
        model.cuda()
        optimizer = Adam(model.parameters(), lr=args.lr, betas=betas, weight_decay=args.weight_decay)
        if args.do_eval:
            load("trm.pt", model)
            test_transformer(model,args,predict_dataloader,"b")
        else:
            train_transformer(model,args,train_dataloader,aug_dataloader0,aug_dataloader1,valid_dataloader,optimizer)
