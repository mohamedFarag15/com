import numpy as np

char_to_idx = {'d': 0, 'o': 1, 'g': 2, 's': 3}
idx_to_char = {i: c for c, i in char_to_idx.items()}

def one_hot(char, vocab_size=4):
    vec = np.zeros((vocab_size, 1))
    vec[char_to_idx[char]] = 1
    return vec

input_chars = ['d', 'o', 'g']
target_char = 's'
x = [one_hot(c) for c in input_chars]
target = one_hot(target_char)

np.random.seed(0)
W_H = np.random.randn(3, 3)
W_X = np.random.randn(3, 4)
W_Y = np.random.randn(4, 3)

h = []
a = []
y = []

h_prev = np.zeros((3, 1))

# Forward Pass 
for t in range(len(x)):
    a_t = np.dot(W_H, h_prev) + np.dot(W_X, x[t])
    h_t = np.tanh(a_t)
    y_t = np.dot(W_Y, h_t)
    y_t = np.exp(y_t) / np.sum(np.exp(y_t)) 

    a.append(a_t)
    h.append(h_t)
    y.append(y_t)
    h_prev = h_t

# Backward Pass
dW_H = np.zeros_like(W_H)
dW_X = np.zeros_like(W_X)
dW_Y = np.zeros_like(W_Y)
dh_next = np.zeros((3, 1))

dy = y[-1] - target
dW_Y += np.dot(dy, h[-1].T)

for t in reversed(range(len(x))):
    dh = np.dot(W_Y.T, dy) if t == 2 else dh_next
    da = dh * (1 - h[t] ** 2)  

    h_prev = np.zeros((3, 1)) if t == 0 else h[t-1]
    
    dW_H += np.dot(da, h_prev.T)
    dW_X += np.dot(da, x[t].T)

    dh_next = np.dot(W_H.T, da)

print("dW_X:\n", dW_X)
print("dW_H:\n", dW_H)
print("dW_Y:\n", dW_Y)
