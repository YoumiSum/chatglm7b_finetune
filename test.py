import transformers
def preprocess(tokenizer, config, example, max_seq_length):
    #问题
    prompt = example["context"]
    #答案
    target = example["target"]
    #问题分词
    prompt_ids = tokenizer.encode(prompt, max_length=max_seq_length, truncation=True)
    #答案分词
    target_ids = tokenizer.encode(
        target,
        max_length=max_seq_length,
        truncation=True,
        add_special_tokens=False)
    input_ids = prompt_ids + target_ids + [config.eos_token_id]
    return input_ids
model_name = "/root/autodl-tmp/models/ChatGLM-6B"
tokenizer = transformers.AutoTokenizer.from_pretrained("/root/autodl-tmp/models/ChatGLM-6B",trust_remote_code=True)
config = transformers.AutoConfig.from_pretrained( model_name, trust_remote_code=True, device_map='auto')
example={"context":"中国的首都是哪？","target":"北京"}
result=preprocess(tokenizer, config, example,10)
print(result)
print([tokenizer.decode(s,skip_special_tokens=False) for s in result])
