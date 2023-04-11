# Machine Translator - German to English

Implementing a Seq2Seq model using PyTorch. The model is trained on the Multi30k dataset, which contains sentence pairs in both German and English. The code has defined three classes, Encoder, Decoder, and Seq2Seq, and it is instantiating objects of these classes to build the model.

The Encoder class has an LSTM layer that takes input in the form of embeddings. The input size to the layer is the size of the German vocabulary, which has been initialized with a Field object from the Torchtext library. The output of the LSTM layer is the hidden and cell states, which are used as the initial state for the Decoder class.

The Decoder class also has an LSTM layer and an output linear layer. The input to the LSTM layer is again an embedding, and the initial state of the layer is the hidden and cell states from the Encoder class. The output of the LSTM layer is passed through the linear layer to obtain the output for each time step. The final output is a tensor of shape (sequence_length, batch_size, output_size).

The Seq2Seq class takes an instance of Encoder and Decoder as input and combines them to create a complete model. The forward method of Seq2Seq takes source and target inputs, which are the German and English sentences, respectively. It passes the source input to the Encoder class to obtain the hidden and cell states. It then uses these states and the target input to generate the output sequence using the Decoder class.

The training loop trains the model for a specified number of epochs using the Adam optimizer and the cross-entropy loss function. The model is trained on batches of the dataset, which are loaded using the BucketIterator from Torchtext. During training, the loss is logged to TensorBoard. The code also defines some hyperparameters for the model, such as the learning rate, batch size, and number of layers in the LSTM layers.
