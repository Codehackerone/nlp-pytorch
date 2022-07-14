# Our own Google Translate

# Imports
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from utils_seq2seq import translate_sentence, bleu, save_checkpoint, load_checkpoint

# Tokenizers
spacy_ger = spacy.load('de')
spacy_eng = spacy.load('en')


def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger, lower=True, init_token='<sos>', eos_token='<eos>')
english = Field(tokenize=tokenizer_eng, lower=True, init_token='<sos>', eos_token='<eos>')

train_data, val_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(german, english))

german.build_vocab(train_data, max_size=10000, min_freq=2)  # freq- Minimum of 2 times to get into vocab
english.build_vocab(train_data, max_size=10000, min_freq=2)


# Encoder LSTM
class Encoder(nn.Module):
    # input-size = size of german vocab
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, drop):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop)

    def forward(self, x):
        # x shape = (seq_length, N)

        embedding = self.dropout(self.embedding(x))
        # embedding shape: (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


# Decoder LSTM
class Decoder(nn.Module):
    # input-size = size of english vocab, output size is equal to input size
    # hidden size of encoder and decoder same
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, drop):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(drop)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=drop)
        self.fc = nn.Linear(hidden_size, output_size)

    # hidden and cell are context vectors from encoder
    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N)
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))
        # embed shape: (1, N, embed_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))
        # shape of outputs: (1, N, hidden_size)

        predictions = self.fc(outputs)
        # shape: (1, N, length_of_vocab) -> sent to loss fn

        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


# Seq2Seq combining Encoder and Decoder
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    # In decoder, as the second predictions depends on first one, we need to use the,
    # correct one sometimes so that seq doesnt turn bad

    # teacher force ratio = juggles between the correct and fake
    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        # (target_len, N)
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        hidden, cell = self.encoder(source)

        # grab start token
        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            # (N, english_vocab_size)
            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


# Ready to train

# Training hyperparamters
num_epochs = 20
learning_rate = 0.001
batch_size = 64

# Model hyperparameters
load_model = False
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)
output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300
hidden_size = 1024
num_layers = 2
enc_dropout = 0.5
dec_dropout = 0.5

# Tensorboard
writer = SummaryWriter(f'runs/loss_plot')
step = 0

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, val_data, test_data),
    batch_size=batch_size,
    sort_within_batch=True,
    sort_key=lambda x:len(x.src), # prioritize similar length sentences.. minimize padding
    device=device
)

encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
decoder_net = Decoder(input_size_decoder, encoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)

model = Seq2Seq(encoder_net, decoder_net).to(device)

pad_idx = english.vocab.stoi['<pad>']
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

if load_model:
    load_checkpoint(torch.load('my_checkpoint.pth.ptar'), model, optimizer)

sentence = 'ich liebe dich von ganzem herzen.'

for epoch in range(num_epochs):
    print(f'Epoch [{epoch} / {num_epochs}]')

    checkpoint = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(model, sentence, german, english, device, max_length=50)
    print(f'Translated sentence:{translated_sentence}')

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        output = model(inp_data, target)
        # Output shape: (trg_len, batch_size, output_dim)

        # (N, 10) and targets will be (N)
        # dont send start token
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()

        writer.add_scalar('Training loss', loss, global_step=step)
        step += 1