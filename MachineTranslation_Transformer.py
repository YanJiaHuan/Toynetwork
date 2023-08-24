### lib ###
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoTokenizer
import datasets

### data ###
dataset_id = 'bentrevett/multi30k'
# English to German translation dataset.
# Train	29,000
# Validation	1,014
# Test	1,000

en_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
ge_tokenizer = AutoTokenizer.from_pretrained('bert-base-german-cased')

print(en_tokenizer(['Hello, my dog is cute. [SEP] I hate dogs']))
print(ge_tokenizer('Hallo, mein Hund ist süß'))
# 这里的attention mask 是告诉模型哪些是padding的，哪些是真实的token，padding的就是0，也代表不需要进行attention计算

## process data ##
def process_fn(examples):
    # Tokenize the texts
    result = en_tokenizer(examples['en'], padding=True, truncation=True,return_tensors='pt',max_length=128)
    result['labels'] = ge_tokenizer(examples['de'], padding=True, truncation=True, return_tensors='pt', max_length=128)['input_ids']
    return result

data = datasets.load_dataset(dataset_id, 'de-en')
data = data.map(process_fn, batched=True)
train_data, val_data, test_data = data['train'], data['validation'], data['test']
# CUDA_VISIBLE_DEVICES=7 python3 MachineTranslation_Transformer.py