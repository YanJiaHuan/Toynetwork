### lib ###
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from transformers import AutoTokenizer
import datasets
import numpy as np
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
        src_seq_length, N = len(src), 1
        trg_seq_length, N = len(trg), 1
        src_mask = src.transpose(0,1) == self.src_pad_idx
        trg_mask = self.transformer.generate_square_subsequent_mask(trg_seq_length).to(self.device)
        src_positions = (torch.arange(0,src_seq_length).unsqueeze(1).expand(src_seq_length,N).to(self.device))
        trg_positions = (torch.arange(0,trg_seq_length).unsqueeze(1).expand(trg_seq_length,N).to(self.device))
        src_embedded = self.dropout(self.src_word_embedding(src) + self.src_position_embedding(src_positions)) # input
        trg_embedded = self.dropout(self.trg_word_embedding(trg) + self.trg_position_embedding(trg_positions)) # target

        out = self.transformer(
            src=src_embedded,
            tgt=trg_embedded,
            src_mask=src_mask,
            tgt_mask=trg_mask
        )
        out = self.fc_out(out)
        return out

### train ###
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
## hyperparameters ##
num_epochs = 10
learning_rate = 3e-4
batch_size = 32
src_vocab_size = en_tokenizer.vocab_size
trg_vocab_size = ge_tokenizer.vocab_size
src_pad_idx = en_tokenizer.pad_token_id
trg_pad_idx = ge_tokenizer.pad_token_id
embed_size = 512
num_heads = 8
num_encoder_layers = 3
num_decoder_layers = 3
dropout = 0.1
max_length = 128
forward_expansion = 4
step = 0
log_step = 100

## process data ##
def process_fn(examples):
    # Tokenize the texts
    result = en_tokenizer(examples['en'], padding=True, truncation=True,max_length=128,return_tensors='pt')
    result['labels'] = ge_tokenizer(examples['de'], padding=True, truncation=True, max_length=128,return_tensors='pt')['input_ids']
    return result

data = datasets.load_dataset(dataset_id, 'de-en')
data = data.map(process_fn, batched=True)
train_data, val_data, test_data = data['train'], data['validation'], data['test']


train_data.set_format(type='torch', columns=['input_ids','attention_mask','labels'])

from torch.utils.data import DataLoader

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

model = Transformer(
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
).to(device)

optimizer = optim.Adam(model.parameters(),lr=learning_rate)
loss_fn = nn.CrossEntropyLoss(ignore_index=trg_pad_idx)

for epoch in range(num_epochs):
    print('Epoch: {}'.format(epoch))
    for i, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        output = model(input_ids,labels[:-1])
        output = output.reshape(-1,output.shape[2])
        labels = labels[1:].reshape(-1)
        optimizer.zero_grad()
        loss = loss_fn(output,labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
        optimizer.step()
        step += 1
        if step % log_step == 0:
            print('Step: {}, Loss: {}'.format(step,loss.item()))




# CUDA_VISIBLE_DEVICES=7 python3 MachineTranslation_Transformer.py