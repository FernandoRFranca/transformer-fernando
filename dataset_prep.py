import torch
from torch.utils.data import Dataset


class TokenizerChar:
    def __init__(self):
        self.chr_to_idx = {chr(v): v for v in range(1, 257)}
        self.chr_to_idx['<SOS>'] = 257
        self.chr_to_idx['<EOS>'] = 258
        self.chr_to_idx['<PAD>'] = 0
        self.chr_to_idx['<UNK>'] = 259

        self.idx_to_chr = {v: k for k, v in self.chr_to_idx.items()}

        self.vocab_size = len(self.chr_to_idx.keys())

    def encode(self, char):
        if char in self.chr_to_idx.keys():
            return self.chr_to_idx[char]
        else:
            return 259
    
    def decode(self, token_idx):
        return self.idx_to_chr[token_idx]
    
    def sos_token(self):
        return '<SOS>'
    
    def sos_token_idx(self):
        return self.chr_to_idx['<SOS>']

    def eos_token(self):
        return '<EOS>'
    
    def eos_token_idx(self):
        return self.chr_to_idx['<EOS>']
    
    def pad_token(self):
        return '<PAD>'
    
    def pad_token_idx(self):
        return self.chr_to_idx['<PAD>']
    
    def get_vocab_size(self):
        return self.vocab_size


class DatasetDialogs(Dataset):
    def __init__(self, dataset_path, sentence_length):
        self.dataset_path = dataset_path
        self.sentence_length = sentence_length
        self.tokenizer = TokenizerChar()

    def __len__(self):
        with open(self.dataset_path, 'r') as dataset:
            num_of_sentences = len(dataset.read().split('\n'))
        return num_of_sentences
    
    def get_shape(self):
        with open(self.dataset_path, 'r') as dataset:
            num_of_sentences = len(dataset.read().split('\n'))
        return (num_of_sentences, self.sentence_length)

    def __getitem__(self, line_idx):
        with open(self.dataset_path, 'r') as dataset:
            selected_sentence = dataset.read().split('\n')[line_idx]
            if len(selected_sentence) < self.sentence_length:
                input_tokens = [self.tokenizer.sos_token_idx()] + [self.tokenizer.encode(char) for char in selected_sentence]
                input_tokens.append(self.tokenizer.eos_token_idx())
                pad_length = self.sentence_length - len(input_tokens) + 1
                pad_tokens = [self.tokenizer.pad_token_idx()] * pad_length
                input_tokens += pad_tokens
            elif len(selected_sentence) == self.sentence_length:
                input_tokens = [self.tokenizer.sos_token_idx()] + [self.tokenizer.encode(char) for char in selected_sentence]
                input_tokens[-1] = self.tokenizer.eos_token_idx()
            elif len(selected_sentence) > self.sentence_length:
                selected_sentence = selected_sentence[:self.sentence_length]
                input_tokens = [self.tokenizer.sos_token_idx()] + [self.tokenizer.encode(char) for char in selected_sentence]
                input_tokens[-1] = self.tokenizer.eos_token_idx()
            # print(f'{len(input_tokens)} - {self.sentence_length}') # debug only
            assert len(input_tokens) == self.sentence_length + 1, f"Lista de índices de tokens não possui mesmo tamanho que 'sentence_length'! len(input_tokens): {len(input_tokens)} - self.sentence_length: {self.sentence_length}"
            try:
                x = torch.tensor(input_tokens[:-1])
                y = torch.tensor(input_tokens[1:])
            except RuntimeError as e:
                print(e)
                print(f"Input tokens: {input_tokens}")
                raise e
        return x, y