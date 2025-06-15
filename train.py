import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch.nn.init as init
from tqdm import tqdm

from dataset_prep import DatasetDialogs
from model import TransformerDecoder


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    sequence_length = 100
    batch_size = 32
    dataset_train = DatasetDialogs('dataset_text/dialogs_train.txt', sequence_length)
    dataset_test = DatasetDialogs('dataset_text/dialogs_test.txt', sequence_length)
    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)
    vocab_size = dataset_train.tokenizer.get_vocab_size()

    d_model = 512
    num_layers = 6
    num_heads = 8
    d_ff = 2048
    dropout = 0.1
    max_seq_length = sequence_length
    model = TransformerDecoder(vocab_size, d_model, max_seq_length, num_layers, d_ff, num_heads, dropout=dropout)  # noqa E501 
    model.to(device)

    tgt_mask = (1 - torch.triu(
    torch.ones(1, sequence_length, sequence_length), diagonal=1)
    ).bool()

    def init_weights(module):
        if isinstance(module, (nn.Linear)):
            init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
            if module.bias is not None:
                init.zeros_(module.bias)
    model.apply(init_weights)

    optimizer = Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    n_epochs = 1000
    n_batches = int(dataset_train.__len__() // batch_size)

    print("Starting model training...")
    for epoch in range(n_epochs):
        print(f"Epoch: {epoch + 1}")
        avg_loss = 0
        model.train()
        for batch_idx, batch in enumerate(tqdm(dataloader_train, total=n_batches)):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x, tgt_mask.to(device))
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))
            avg_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        torch.save(model.state_dict(), f'model_checkpoints/model_checkpoint_{epoch+1}.pth')

        avg_loss /= (batch_idx + 1)
        print(f"Average epoch training loss: {avg_loss}")
        print(f"Last batch training loss: {loss}")

        model.eval()
        avg_loss = 0
        for batch_idx, batch in enumerate(dataloader_test):
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x, tgt_mask.to(device))
            loss = loss_fn(outputs.view(-1, vocab_size), y.view(-1))
            avg_loss += loss.item()
        
        avg_loss /= (batch_idx + 1)
        print(f"Epoch validation loss: {avg_loss}")
