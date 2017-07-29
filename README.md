# BasicRNN
As the name suggests, this is just a simple, vanilla Recurrent Neural Network, written in Python/Theano.

I used the Bible to train it as it was just an easy text to find.

## Training
I trained the RNN for 50 epochs, 20k sentences, 100-size hidden layer, 0.005 initial learning rate, 5 epochs per loss evaluation.
  
`THEANO_FLAGS='device=cuda,floatX=float32' nohup python -u rnn.py 20000 0.005 50 5 > rnn.log &`

## Text generation
You can find example generated sentences in logs/sentences.log. They aren't that great right now. They don't make any sense (if they do, it was by chance) as basic RNNs aren't supposed to make sense until you so-called upgrade them to an LSTM or something of the sort. In an attempt to reduce the number of duplicate sentences, I used a shrinking resampling probability interval where any word sampled with a probability in that range would be discarded. The range shrinks as the sentence grows.

## Inspiration
Credit where credit is due. This code follows closely with the setup [here](http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-1-introduction-to-rnns/).
