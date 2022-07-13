# STEPS:

# 1. Specify how preprocessing should be done -> Fields
# 2. Use Dataset to load the data -> TabularDataset (JSON/CSV/TSV Files)
# 3. Contruct an iterator to do batching and padding -> BucketIterator


from torchtext.data import Field, TabularDataset,BucketIterator
import spacy

spacy_en = spacy.load('en')

def tokenize(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]

tokenize = lambda x:x.split()

quote = Field(sequential=True, use_vocab=True, tokenize=tokenize, lower=True)
score = Field(sequential=False, use_vocab=False)

fields = {'quote':('q', quote), 'score':('s', score)}

train_data, test_data = TabularDataset.splits(
    path='data',
    train='train.json',
    test='test.json',
    #validation='validation.json'
    format='json',
    fields=fields
)

# train_data, test_data = TabularDataset.splits(
#     path='data',
#     train='train.csv',
#     test='test.csv',
#     format='json',
#     fields=fields
# )
#
# train_data, test_data = TabularDataset.splits(
#     path='data',
#     train='train.csv',
#     test='test.csv',
#     format='json',
#     fields=fields
# )


# print(train_data[0].__dict__.keys())
# print(train_data[0].__dict__.values())
#

quote.build_vocab(train_data, max_size=10000, min_freq=1
                  # If pretrained
                  # vectors='glove.6B.100d'
                  )

train_iterator, test_iterator = BucketIterator.splits(
    (train_data, test_data),
    batch_size=2,
    #device='cuda'
)

for batch in train_iterator:
    print(batch.q)
    print(batch.s)

