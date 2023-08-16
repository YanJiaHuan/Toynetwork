### lib ###
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext import data, datasets
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter

### load data ###
import en_core_web_sm
import de_core_news_sm
spacy_ger = de_core_news_sm.load()
spacy_eng = en_core_web_sm.load()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
### help functions ###
sos_tok = '<sos>'
eos_tok = '<eos>'
def tokenize_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]

def tokenize_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]

text = 'I like apples'
doc = spacy_eng(text)
for token in doc:
    print(token.text, token.pos_, token.dep_)

train_data, valid_data, test_data = Multi30k(root='.data',split=('train', 'valid', 'test'), language_pair=('de', 'en'))


class TextDatasets(torch.utils.data.Dataset):
    def __init__(self, raw_data):
        self.datasets = list(raw_data)

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        src, trg = self.datasets[idx]
        src = [sos_tok] + tokenize_ger(src) + [eos_tok]
        trg = [sos_tok] + tokenize_eng(trg) + [eos_tok]
        return src, trg


train_datasets = TextDatasets(train_data)
valid_datasets = TextDatasets(valid_data)
test_datasets = TextDatasets(test_data)

print("Train size:", len(train_datasets))
print("Valid size:", len(valid_datasets))
print("Test size:", len(test_datasets))

### network ###
class Encoder(nn.Module):
    def __init__(self,input_size,embedding_size,hidden_size,num_layers,p):
        super(Encoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
    def forward(self,x):
        # x shape: (seq_length, N) where N is batch size
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding)
        return hidden, cell
class Decoder(nn.Module):
    def __init__(self,input_size, embedding_size, hidden_size, output_size, num_layers, p):
        super(Decoder,self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)
        self.embedding = nn.Embedding(input_size,embedding_size)
        self.rnn = nn.LSTM(embedding_size,hidden_size,num_layers,dropout=p)
        self.fc = nn.Linear(hidden_size,output_size)
    def forward(self,x,hidden,cell):
# shape of x: (N) but we want (1,N)
        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))
        # embedding shape: (1,N,embedding_size)
        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1,N,hidden_size)
        predictions = self.fc(outputs)
        # shape of predictions: (1,N,length_of_vocab)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell

class seq2seq(nn.Module):
    def __init__(self,encoder,decoder):
        super(seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self,source,target,teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(german.vocab)
        outputs = torch.zeros(target_len,batch_size,target_vocab_size).to(device)
        hidden, cell = self.encoder(source)
        x = target[0] # get the first token
        for t in range(1,target_len):
            output, hidden, cell = self.decoder(x,hidden,cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random() < teacher_force_ratio else best_guess
        return outputs

### training ###
## hyperparameters ##
num_epochs = 20
learning_rate = 1e-3
batch_size = 64

## model parameters ##
load_model = False
device = device
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

## tensorboard ##
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x: len(x.src),
    device=device)
encoder_net = Encoder(input_size_encoder,encoder_embedding_size,hidden_size,num_layers,enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder,decoder_embedding_size,hidden_size,output_size,num_layers,dec_dropout).to(device)

model = seq2seq(encoder_net,decoder_net).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi['<pad>']
loss_fn = nn.CrossEntropyLoss(ignore_index=pad_idx)

test_sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."
# a boat with several men on it is pulled ashore by a large team of horses.
for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    model.eval()
    translated_sentence = translate_sentence(model, test_sentence, german, english, device, max_length=50)
    print('Translated example sentence: \n', translated_sentence)
    for batch_idx, batch in enumerate(train_iterator):
        input_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(input_data, target)
        # output shape: (trg_len, batch_size, output_dim)
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = loss_fn(output, target)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1










