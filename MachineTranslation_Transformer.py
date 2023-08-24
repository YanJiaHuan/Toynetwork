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

# print(en_tokenizer(['Hello, my dog is cute. [SEP] I hate dogs']))
# print(ge_tokenizer('Hallo, mein Hund ist süß'))
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
# print(train_data[0])

class Transformer(nn.Module):
    def __init__(
            self,
            src_vocab_size,
            trg_vocab_size,
            src_pad_idx,
            trg_pad_idx,
            device,
            embed_size,
            num_encoder_layers,
            num_decoder_layers,
            forward_expansion,
            num_heads,
            dropout,
            max_length
    ):
        super(Transformer,self).__init__()
        self.src_word_embedding = nn.Embedding(src_vocab_size,embed_size)
        self.src_position_embedding = nn.Embedding(max_length,embed_size)
        self.trg_word_embedding = nn.Embedding(trg_vocab_size,embed_size)
        self.trg_position_embedding = nn.Embedding(max_length,embed_size)
        self.device = device
        self.transformer = nn.Transformer(
            d_model=embed_size,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=forward_expansion*embed_size,
            dropout=dropout
        )
        self.fc_out = nn.Linear(embed_size,trg_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.max_length = max_length

    def forward(self,src,trg):
        src_seq_length, N = src.shape
        trg_seq_length, N = trg.






# CUDA_VISIBLE_DEVICES=7 python3 MachineTranslation_Transformer.py