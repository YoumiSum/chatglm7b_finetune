from transformers.integrations import TensorBoardCallback
from torch.utils.tensorboard import SummaryWriter
from transformers import TrainingArguments,TrainerCallback
from transformers import Trainer, HfArgumentParser
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn as nn
from peft import get_peft_model, LoraConfig, TaskType
from dataclasses import dataclass, field
import datasets
import os
import glob
import shutil
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn

base_model = "/root/autodl-tmp/models/ChatGLM-6B"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

@dataclass
class FinetuneArguments:
    dataset_path: str = field(default="data/alpaca")
    model_path: str = field(default="output")
    lora_rank: int = field(default=8)


class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)


def data_collator(features: list) -> dict:
    len_ids = [len(feature["input_ids"]) for feature in features]
    longest = max(len_ids)
    input_ids = []
    labels_list = []
    for ids_l, feature in sorted(zip(len_ids, features), key=lambda x: -x[0]):
        #问题+答案
        ids = feature["input_ids"]
        #问题长度
        seq_len = feature["seq_len"]
        #-100特殊字符，表示不预测
        # [-100] * (seq_len - 1) 问题部分是不需要预测的
        #ids[(seq_len - 1) :] 预测答案
        #[-100] * (longest - ids_l)  不零位置不需要预测
        labels = (
            [-100] * (seq_len - 1) + ids[(seq_len - 1) :] + [-100] * (longest - ids_l)
        )
        ids = ids + [tokenizer.pad_token_id] * (longest - ids_l)
        _ids = torch.LongTensor(ids)
        labels_list.append(torch.LongTensor(labels))
        input_ids.append(_ids)
    input_ids = torch.stack(input_ids)
    labels = torch.stack(labels_list)
    return {
        "input_ids": input_ids,
        "labels": labels,
    }


class ModifiedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        return model(
            input_ids=inputs["input_ids"],
            labels=inputs["labels"],
        ).loss

    def save_model(self, output_dir=None, _internal_call=False):
        from transformers.trainer import TRAINING_ARGS_NAME

        os.makedirs(output_dir, exist_ok=True)
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        saved_params = {
            k: v.to("cpu") for k, v in self.model.named_parameters() if v.requires_grad
        }
        torch.save(saved_params, os.path.join(output_dir, "adapter_model.bin"))




class Adapter(nn.Module):
    """
    外挂（适配器）模型
    """
    def __init__(self, in_features, mid_features):
        super(Adapter, self).__init__() # or nn.Module.__init__(self)
        self.w1 = nn.Linear(in_features, mid_features)
        self.w2 = nn.Linear(mid_features, in_features)
        self.act=nn.ReLU()


    def forward(self, x):
        y = self.w1(x)
        y=self.act(y)
        y = self.w2(y)
        return 1e-1*y + x
    
    
class CombinedModel(nn.Module):
    """
    把适配器和原有的全连接层绑在一起
    """
    def __init__(self, submodel1, submodel2):
        super(CombinedModel, self).__init__()
        #需要训练的，都需要使用float32
        self.submodel1 = submodel1.to(torch.float32)
        self.submodel2 = submodel2.to(torch.float32)
 
    def forward(self, x):
        x = x.to(torch.float32)
        y1 = self.submodel1(x)
        y2 = self.submodel2(y1)
        return y2


def get_trainable_para_num(model):
    lst = []
    for para in model.parameters():
        if para.requires_grad == True:
            lst.append(para.nelement())
    print(f"trainable paras number: {sum(lst)}")


def train(base_model="/root/autodl-tmp/models/ChatGLM-6B", data_path="data/wenlv_token", log_dir="/root/tf-logs",
         add_layes=[0], 
         output_dir="chatglm-6b-adapter",
         per_device_train_batch_size=10,
         remove_unused_columns=False,
         num_train_epochs=1,
         learning_rate=1e-5,
         adapter_mid_features=4):
    """
    :param log_dir: 日志存储路径，注意：会删除该目录下的所有内容
    :param base_model: 基础模型路径
    :param data_path: 数据存放路径
    :param add_layes: 要在哪些层中增加add_layers，[0]表示只在第0层添加，[0, 1]表示在第0和第1层添加
        可以使用print(model) 来查看有多少层
        例如：下面是一个chatglm_6b的模型架构
        ##########################################################################################################################
        ChatGLMForConditionalGeneration(
            (transformer): ChatGLMModel(
                (word_embeddings): Embedding(130528, 4096)
                (layers): ModuleList(
                (0-27): 28 x GLMBlock(
                    (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
                    (attention): SelfAttention(
                    (rotary_emb): RotaryEmbedding()
                    (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
                    (dense): Linear(in_features=4096, out_features=4096, bias=True)
                    )
                    (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
                    (mlp): GLU(
                    (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
                    (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
                    )
                )
                )
                (final_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
            )
            (lm_head): Linear(in_features=4096, out_features=130528, bias=False)
        )
        ##########################################################################################################################
        可以看到GLMBlock有(0-27)层，所以add_layes的取值范围是[0, 27]

    """

    os.environ["WANDB_DISABLED"] = "true"
    if os.path.exists(log_dir): shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir)
    training_args = TrainingArguments(output_dir=output_dir, 
                                      per_device_train_batch_size=per_device_train_batch_size,
                                      remove_unused_columns=remove_unused_columns,
                                      num_train_epochs=num_train_epochs,
                                      learning_rate=learning_rate)
    # # init model
  
    #加载训练好的模型模型
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True,device_map="auto").cuda().to(torch.float32)

    #冻结住模型的所有参数
    for name, param in model.named_parameters():
        param.requires_grad=False
    
    # 加入外挂模型
    adapter_list = {}
    for idx, param in enumerate(model.transformer.layers):
        if idx in add_layes:
            dense = param.attention.dense
            in_features = dense.out_features

            adapter = Adapter(in_features=in_features, mid_features=adapter_mid_features)
            combind = CombinedModel(dense, adapter)
            param.attention.dense = combind

            adapter_list[idx] = adapter
     
    # 显示加入外挂后的模型信息
    print("##########################################################################################################################")
    print(model)
    print("##########################################################################################################################")
    get_trainable_para_num(model)

    dataset = datasets.load_from_disk(data_path)
    print(dataset)
    print(f"\n{len(dataset)=}\n")
    trainer = ModifiedTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        callbacks=[TensorBoardCallback(writer)],
        data_collator=data_collator,
    )
    #不能这样直接保存，直接保存不会报错
    #但是加载不起来，因为没有改配置文件
    #model.save_pretrained(training_args.output_dir)
    trainer.train()
    writer.close()
     
    for i, adapter in adapter_list.items():
        torch.save(adapter, training_args.output_dir + "/" + str(i))
    

def load_model(base_model="/root/autodl-tmp/models/ChatGLM-6B",
               adapter_dir="chatglm-6b-adapter"):
    model = AutoModel.from_pretrained(base_model, trust_remote_code=True,device_map="auto").cuda().to(torch.float32)

    files = glob.glob(os.path.join(adapter_dir, '*'))
    
    # 只保留文件名只包含数字的文件
    numeric_files = [os.path.basename(file) for file in files if os.path.isfile(file) and os.path.basename(file).isdigit()]
    adapter_list = {}
    for i in numeric_files:
        i = int(i)
        adapter_list[i] = torch.load("chatglm-6b-adapter/{}".format(i)).cuda()

    for idx, param in enumerate(model.transformer.layers):
        if idx in adapter_list.keys():
            dense = param.attention.dense

            adapter = adapter_list[idx]
            combind = CombinedModel(dense, adapter)
            param.attention.dense = combind
    
    # 显示加入外挂后的模型信息
    print("##########################################################################################################################")
    print(model)
    print("##########################################################################################################################")

    return model


if __name__ == "__main__":
    
    train(base_model=base_model, data_path="data/wenlv_token", 
         add_layes=[0], 
         output_dir="chatglm-6b-adapter",
         per_device_train_batch_size=4,
         remove_unused_columns=False,
         num_train_epochs=50,
         learning_rate=1e-5
    )

    # model = load_model(base_model="/root/autodl-tmp/models/ChatGLM-6B",
    #            adapter_dir="chatglm-6b-adapter")
    # model = model.half()
    # inp = "自驾游从九江到云南版纳怎么走?"
    # response, history = model.chat(tokenizer, inp, history=[],max_length=250)
    # print(response)
