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

print(en_tokenizer('Hello, my dog is cute'))
print(ge_tokenizer('Hallo, mein Hund ist süß'))
# CUDA_VISIBLE_DEVICES=7 python3 MachineTranslation_Transformer.py