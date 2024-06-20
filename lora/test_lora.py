
from transformers import AutoTokenizer, AutoModel
from peft import PeftModel

from peft import PrefixEncoder, PrefixTuningConfig


base_model = "/root/autodl-tmp/models/ChatGLM-6B"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
model = AutoModel.from_pretrained(base_model, trust_remote_code=True).cuda()
 
# 加载lora
model = PeftModel.from_pretrained(model, "/root/autodl-tmp/sources/01/lora/wenlv_lora").half()
model=model.eval()
print(model)

# inp="Instruction: 你是谁\nAnswer: "
# response, history = model.chat(tokenizer, inp, history=[])
# print(response)

inp = "自驾游从九江到云南版纳怎么走?"
response, history = model.chat(tokenizer, inp, history=[], max_length=250)
print(response)

