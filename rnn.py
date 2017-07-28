import nltk
import itertools
import numpy as np
from datetime import datetime
import theano as theano
import theano.tensor as T
import sys
import operator
import random
import pickle
import os
from scipy.misc import logsumexp
from math import log

VOCAB_SIZE = 8000
SENTENCE_START = "SENTENCE_START"
SENTENCE_END = "SENTENCE_END"
UNKNOWN_TOKEN = "UNKNOWN_TOKEN"

# Tokenize bible text into words in sentences
def buildVocabulary(loadIfThere=False):
    if loadIfThere:
        if os.path.isfile('vocab_word_by_index.pickle') and os.path.isfile('vocab_index_by_word.pickle'):
            with open('vocab_word_by_index.pickle', 'rb') as f:
                vocab_word_by_index = pickle.load(f)
                with open('vocab_index_by_word.pickle', 'rb') as f:
                    vocab_index_by_word = pickle.load(f)
                    with open('tokens.pickle', 'rb') as f:
                        tokens = pickle.load(f)
                        return vocab_word_by_index, vocab_index_by_word, tokens
    
    print('Tokenizing...')
    tokens = []
    with open('bible.txt', 'r') as f:
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
        
    # Randomize training set
    random.shuffle(tokens)
    
    # Save vocab
    with open('vocab_word_by_index.pickle', 'wb') as f:
        pickle.dump(vocab_word_by_index, f)
    with open('vocab_index_by_word.pickle', 'wb') as f:
        pickle.dump(vocab_index_by_word, f)
    with open('tokens.pickle', 'wb') as f:
        pickle.dump(tokens, f)

    return vocab_word_by_index, vocab_index_by_word, tokens

def formatTrainingData(vocab_index_by_word, tokens):
    print('Formatting training set...')
    # Format training data
    x_train = np.asarray([[vocab_index_by_word[w] for w in sentence[:-1]] for sentence in tokens])
    y_train = np.asarray([[vocab_index_by_word[w] for w in sentence[1:]] for sentence in tokens])
    return x_train, y_train

def translate(vocab_word_by_index, arr):
    #return vocab_word_by_index[ind]
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
            s_t = T.tanh(U[:,x_t] + W.dot(s_t_prev)) + 0.01*s_t_prev
            o_t = T.nnet.softmax(V.dot(s_t))
            #premax_o = V.dot(s_t)
            #o_t = np.exp(premax_o - logsumexp(premax_o))
            return [o_t[0], s_t]
            #return [o_t, s_t]

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
        return self.summedLoss(X, Y) / float(np.sum([len(y) for y in Y])) 

    def summedLoss(self, X, Y):
        if len(X) == 0 or len(Y) == 0: return 0
        return np.sum([self.calculateError(x,y) for x,y in zip(X,Y)])

def gradientCheck(model, x, y, h=0.001, errorThreshold=0.01):
    model.bptt_truncate = 1000
    bptt_gradients = model.bptt(x, y)
    model_parameters = ['U', 'V', 'W']
    for pidx, pname in enumerate(model_parameters):
        parameter_T = operator.attrgetter(pname)(model)
        parameter = parameter_T.get_value()
        print("Performing gradient check for parameter " + str(pname) + " with size " + str(np.prod(parameter.shape)) + ".")
        it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            ix = it.multi_index
            original_value = parameter[ix]
            parameter[ix] = original_value + h
            parameter_T.set_value(parameter)
            gradplus = model.summedLoss([x],[y])
            parameter[ix] = original_value - h
            parameter_T.set_value(parameter)
            gradminus = model.summedLoss([x],[y])
            estimated_gradient = (gradplus - gradminus)/(2*h)
            parameter[ix] = original_value
            parameter_T.set_value(parameter)
            
            backprop_gradient = bptt_gradients[pidx][ix]
            relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
            if relative_error > errorThreshold:
                print("Gradient Check ERROR: parameter=" + str(pname) + " ix=" + str(ix))
                print("+h Loss: " + str(gradplus))
                print("-h Loss: " + str(gradminus))
                print("Estimated gradient: " + str(estimated_gradient))
                print("Backpropagation gradient: " + str(backprop_gradient))
                print("Relative error: " + str(relative_error))
                return
            it.iternext()
        print("Gradient check for parameter " + str(pname) + " passed.")

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

def generateText(model, vocab_index_by_word, vocab_word_by_index, sentenceMaxLength=15, nSentences=10):
    generated = []
    for n in range(nSentences):
        sentence = [vocab_index_by_word[SENTENCE_START]]
        for i in range(sentenceMaxLength):
            nextWordVec = model.forwardPropagation(sentence)
            sampleWord = vocab_index_by_word[UNKNOWN_TOKEN]
            #while vocab_index_by_word[UNKNOWN_TOKEN] == sampleWord or (nextWordVec[-1][sampleWord] > (0.25/(i+1)) and nextWordVec[-1][sampleWord] < (0.75-0.25/(i+1))):
            while vocab_index_by_word[UNKNOWN_TOKEN] == sampleWord or (nextWordVec[-1][sampleWord] > (0.25 + 0.05*log(i+1)) and nextWordVec[-1][sampleWord] < (0.5-0.05*log(i+1))):
                choices = range(len(nextWordVec[-1]))
                sampleWord = np.random.choice(choices, p=nextWordVec[-1])
                #while np.sum(nextWordVec[-1]) > 1-1e-9:
                #    nextWordVec[-1] /= (1+1e-5)
                #sampleWord = np.argmax(np.random.multinomial(1,nextWordVec[-1]))
            sentence.append(sampleWord)
            if sampleWord == vocab_index_by_word[SENTENCE_END]:
                break
        generated.append(' '.join(translate(vocab_word_by_index, sentence[1:-1])))
    return generated

def getRNNModel(loadIfThere=False):
    if loadIfThere:
        if os.path.isfile('RNNmodel.pickle'):
            with open('RNNmodel.pickle', 'rb') as f:
                model = pickle.load(f)
                return model
    model = RNN(VOCAB_SIZE)
    return model

def main():
    sys.stdout.flush()
    #np.random.seed(0)
    vocab_word_by_index, vocab_index_by_word, tokens = buildVocabulary(True)
    x_train, y_train = formatTrainingData(vocab_index_by_word, tokens)
    model = getRNNModel(True)
    if len(sys.argv) == 5:
        _trainingSize = min(len(x_train), int(sys.argv[1]))
        _learningRate = float(sys.argv[2])
        _nepoch = int(sys.argv[3])
        _evaluateLossAfter = int(sys.argv[4])
        model.loss(x_train[:5000], y_train[:5000])
        trainWithSgd(model, x_train[:_trainingSize], y_train[:_trainingSize], learningRate=_learningRate, nepoch=_nepoch, evaluateLossAfter=_evaluateLossAfter)
        with open('RNNmodel.pickle', 'wb') as f:
            pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    generatedSentences = generateText(model, vocab_index_by_word, vocab_word_by_index, 50, 25)
    for gSent in generatedSentences:
        print(gSent)
main()
