### lib ###
import io
import os
import unicodedata
import string
import glob
import torch
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


### model ###
