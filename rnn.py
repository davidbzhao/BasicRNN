import nltk
import itertools
import numpy as np
from datetime import datetime
import theano
import theano.tensor as T
import sys


VOCAB_SIZE = 8000
SENTENCE_START = "SENTENCE_START"
SENTENCE_END = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

# Tokenize bible text into words in sentences
print('Tokenizing...')
tokens = []
with open('bible.txt') as f:
    string = ''
    for line in f:
        if line.isspace():
            sentences = nltk.sent_tokenize(string)
            for s in sentences:
                words = [w.lower() for w in nltk.word_tokenize(s)]
                tokens.append([SENTENCE_START] + words + [SENTENCE_END])
            string = ''
        else:
            string += line
    if string != '':
        tokens.append(nltk.sent_tokenize(string))

# Build a vocabulary of the 8000 most common words
print('Assembling vocabulary...')
word_frequency = nltk.FreqDist(itertools.chain(*tokens))
vocab = word_frequency.most_common(VOCAB_SIZE - 1)

vocab_frequencies = dict(vocab)
vocab_word_by_index = [w for w, f in vocab]
vocab_word_by_index.append(UNKNOWN_TOKEN)
vocab_index_by_word = dict([(w,i) for i,w in enumerate(vocab_word_by_index)])

# Replace words not in vocab with UNKNOWN_TOKEN
for i, sentence in enumerate(tokens):
    tokens[i] = [w if w in vocab_index_by_word else UNKNOWN_TOKEN for w in sentence]

print('Formatting training set...')
# Format training data
x_train = np.asarray([[vocab_index_by_word[w] for w in sentence[:-1]] for sentence in tokens])
y_train = np.asarray([[vocab_index_by_word[w] for w in sentence[1:]] for sentence in tokens])

def translate(arr):
    return [vocab_word_by_index[arr[i]] for i in range(len(arr))]

class RNN:
    def __init__(self, vocab_size, hidden_dim=100, bptt_truncate=4):
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        U = np.random.uniform(-np.sqrt(1./vocab_size), np.sqrt(1./vocab_size), (hidden_dim, vocab_size))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (vocab_size, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.theano = {}
        self.__theano_build__()

    def __theano_build__(self):
        U = self.U
        V = self.V
        W = self.W
        x = T.ivector('x')
        y = T.ivector('y')

        def forwardPropagationStep(x_t, s_t_prev, U, V, W):
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev))
            o_t = T.nnet.softmax(V.dot(s_t))[0]
            return [o_t, s_t]

        [o,s], updates = theano.scan(
            forwardPropagationStep,
            sequences=x,
            non_sequences=[U,V,W],
            outputs_info=[None, dict(initial=T.zeros(self.hidden_dim))],
            truncate_gradient=self.bptt_truncate,
            strict=True)

        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o,y))

        dU = T.grad(o_error, U)
        dV = T.grad(o_error, V)
        dW = T.grad(o_error, W)
    
        self.forwardPropagation = theano.function([x], o)
        self.predict = theano.function([x], prediction)
        self.calculateError = theano.function([x, y], o_error)
        self.bptt = theano.function([x, y], [dU, dV, dW])

        learningRate = T.scalar('learningRate')
        self.sgd_step = theano.function(
            [x,y,learningRate],
            [],
            updates=[(self.U, self.U - learningRate * dU),
                    (self.V, self.V - learningRate * dV),
                    (self.W, self.W - learningRate * dW)]
        )
    def loss(self, X, Y):
        return np.sum([self.calculateError(x,y) for x,y in zip(X,Y)]) / float(np.sum([len(y) for y in Y])) 

def trainWithSgd(model, X_train, y_train, learningRate=0.005, nepoch=1, evaluateLossAfter=5):
    # We keep track of the losses so we can plot them later
    losses = []
    num_examples_seen = 0
    for epoch in range(nepoch):
        # Optionally evaluate the loss
        if (epoch % evaluateLossAfter == 0):
            loss = model.loss(X_train, y_train)
            losses.append((num_examples_seen, loss))
            time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            print(str(time) + ": Loss after num_examples_seen=" + str(num_examples_seen) + " epoch=" + str(epoch) + ": " + str(loss))
            # Adjust the learning rate if loss increases
            if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                learningRate = learningRate * 0.5  
                print("Setting learning rate to " + str(learningRate))
            sys.stdout.flush()
        # For each training example...
        for i in range(len(y_train)):
            model.sgd_step(X_train[i], y_train[i], learningRate)
            num_examples_seen += 1

np.random.seed(10)
model = RNN(VOCAB_SIZE)
print('Expected loss', end=' : ')
print(np.log(VOCAB_SIZE))
print('Model loss', end=' : ')
print(model.loss(x_train[:1000], y_train[:1000]))
trainWithSgd(model, x_train[:1000], y_train[:1000], nepoch=2, evaluateLossAfter=1)