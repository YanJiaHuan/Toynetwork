### lib ###
import io
import os
import unicodedata
import string
import glob
import torch
import torch.nn as nn
import matplotlib as plt
import random

### utils ###
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

def unicode_to_ascii(s):
    return ''.join(
            c for c in unicodedata.normalize('NFD',s)
            if unicodedata.category(c) != 'Mn'
            and c in ALL_LETTERS
            )
def load_data(path):
    category_lines = {}
    all_categories = []

    def find_files(path):
        return glob.glob(path)
    def read_lines(filename):
        lines = io.open(filename, encoding = 'utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    for filename in find_files(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = read_lines(filename)
        category_lines[category] = lines

    return category_lines, all_categories
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)
def letter_to_tensor(letter):
    tensor = torch.zeros(1,N_LETTERS)
    tensor[0][letter_to_index(letter)] = 1
    return tensor
def line_to_tensor(line):
    tensor = torch.zeros(len(line),1,N_LETTERS)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
def random_training_example(category_lines, all_categories):
    def random_choice(a):
        random_idx = random.randint(0,len(a)-1)
        return a[random_idx]
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype = torch.long)
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

### model ###

class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size):
        super(RNN,self).__init__()

        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size) # input to hidden
        self.i2o = nn.Linear(input_size + hidden_size, output_size) # input to output
        self.softmax = nn.LogSoftmax(dim = 1)
    def forward(self, input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)

category_lines, all_categories = load_data('data/names/*.txt') # 读取数据
print(f'There are {len(all_categories)} categories.')

## help function ##
def category_from_output(output):
    category_id = torch.argmax(output).item()
    return all_categories[category_id]

### training ###
## setup ##
loss_fn = nn.CrossEntropyLoss()
learning_rate = 1e-3
n_hidden = 128
model = RNN(N_LETTERS, len(all_categories), n_hidden)
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

def train(line_tensor, category_tensor):
    hidden = model.init_hidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = model(line_tensor[i],hidden)

    loss = loss_fn(output, category_tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return output, loss.item()

current_loss = 0
all_losses = []
plot_steps, print_steps = 1000, 5000
n_iters = 100000
for i in range(n_iters):
    category, line, category_tensor, line_tensor = random_training_example(category_lines, all_categories)
    output, loss = train(line_tensor, category_tensor)
    current_loss += loss

    if (i+1) % plot_steps == 0:
        all_losses.append(current_loss/plot_steps)
        current_loss = 0

    if (i+1) % print_steps == 0:
        guess = category_from_output(output)
        correct = '✓' if guess == category else '✗ (%s)' % category
        print(f'{i+1} {(i+1)/n_iters*100} {loss:.4f} {line} / {guess} {correct}')
plt.figure()
plt.plot(all_losses)
plt.show()

def predict(line_input):
    print(f'\n> {line_input}')
    with torch.no_grad():
        line_tenosr = line_to_tensor(line_input)
        hidden = model.init_hidden()
        for i in range(line_tenosr.size()[0]):
            output, hidden = model(line_tenosr[i], hidden)
        guess = category_from_output(output)
        return guess

## save ##
torch.save(model.state_dict(), "Model/model.pth")






