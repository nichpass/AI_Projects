# from video: https://www.youtube.com/watch?v=BwmddtPFWtA
import matplotlib.pyplot as plt
import numpy as np
data = open('kafka.txt', 'r').read()
plt_data = []

chars = list(set(data))
data_size, vocab_size = len(data), len(chars)

# char to int and int to char dictionaries
char_to_ix = {ch: i for i, ch in enumerate(chars)}
ix_to_char = {i: ch for i, ch in enumerate(chars)}


# hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model parameters
wxh = np.random.randn(hidden_size, vocab_size) * 0.01 # input to hidden state
bh = np.zeros((hidden_size, 1))

whh = np.random.randn(hidden_size, hidden_size) * 0.01 # input to hidden state
by = np.zeros((vocab_size, 1))

why = np.random.randn(vocab_size, hidden_size) * 0.01 # input to hidden state


def lossFunction(inputs, targets, hprev):
    # dictionaries for the input, hidden, target, and calculated probabilities
    xs, hs, ys, ps = {}, {}, {}, {}

    # want separate memory not reference to original array
    hs[-1] = np.copy(hprev)

    # init loss
    loss = 0

    # forward pass
    for t in range(len(inputs)):
        xs[t] = np.zeros((vocab_size, 1))
        xs[t][inputs[t]] = 1
        hs[t] = np.tanh(np.dot(wxh, xs[t]) + np.dot(whh, hs[t-1]) + bh)
        ys[t] = np.dot(why, hs[t]) + by

        ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t]))
        loss += - np.log(ps[t][targets[t], 0])

    # backward pass: get the gradients
    dwxh, dwhh, dwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
    dbh, dby = np.zeros_like(bh), np.zeros_like(by)
    dhnext = np.zeros_like(hs[0])

    for t in reversed(range(len(inputs))):
        dy = np.copy(ps[t])
        dy[targets[t]] -= 1

        dwhy += np.dot(dy, hs[t].T)
        dby += dy

        dh = np.dot(why.T, dy) + dhnext
        dhraw = (1 - hs[t] * hs[t]) * dh
        dbh += dhraw
        dwxh += np.dot(dhraw, xs[t].T)
        dwhh += np.dot(dhraw, hs[t-1].T)
        dhnext = np.dot(whh.T, dhraw)

    for dparam in [dwxh, dwhh, dwhy, dbh, dby]:
        np.clip(dparam, -5, 5, out=dparam)
    return loss, dwxh, dwhh, dwhy, dbh, dby, hs[len(inputs)-1]

# create a sentence from the model
def sample(h, seed_ix, n):
    x = np.zeros((vocab_size, 1))
    x[seed_ix] = 1
    ixes = []

    for t in range(n):
        h = np.tanh(np.dot(wxh, x) + np.dot(whh, h) + bh)

        y = np.dot(why, h) + by
        p = np.exp(y) / np.sum(np.exp(y))

        ix = np.random.choice(range(vocab_size), p=p.ravel())
        x = np.zeros((vocab_size, 1))
        x[ix] = 1
        ixes.append(ix)

    txt = ''.join(ix_to_char[ix] for ix in ixes)
    print('----\n %s \n----' % (txt, ))


n, p = 0, 0
mwxh, mwhh, mwhy = np.zeros_like(wxh), np.zeros_like(whh), np.zeros_like(why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

# main training loop
while n <= 1000*100:
    if p+seq_length+1 >= len(data) or n == 0:
        hprev = np.zeros((hidden_size, 1)) # reset rnn memory
        p = 0 # go from start of data

    inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
    targets = [char_to_ix[ch] for ch in data[p+1:p + seq_length+1]]

    # forward seq_length characters through the net and fetch gradient
    loss, dwxh, dwhh, dwhy, dbh, dby, hprev = lossFunction(inputs, targets, hprev)
    smooth_loss = smooth_loss * 0.999 + loss * 0.001

    # sample from model now and then
    if n % 1000 == 0:
        print('iter %d, loss %f' % (n, smooth_loss)) # print progress
        plt_data.append(smooth_loss)
        sample(hprev, inputs[0], 200)

    # perform parameter update with adagrad
    for param, dparam, mem in zip([wxh, whh, why, bh, by],
                                  [dwxh, dwhh, dwhy, dbh, dby],
                                  [mwxh, mwhh, mwhy, mbh, mby]):
        mem += dparam ** 2
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
    p += seq_length # move data pointer
    n += 1 # iteration counter
plt.plot(plt_data)
plt.show()