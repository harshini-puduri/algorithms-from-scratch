import numpy as np
import pandas as pd

corpus = [
    "I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation.",
    "Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation.",
    "This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice.",
    "It came as a joyous daybreak to end the long night of their captivity.",
    "But one hundred years later, the Negro still is not free.",
    "One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination.",
    "One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity.",
    "One hundred years later, the Negro is still languishing in the corners of American society and finds himself an exile in his own land."
]

tokenized_corpus = [doc.lower().split() for doc in corpus]
vocab = set([word for sentence in tokenized_corpus for word in sentence])
word2idx = {word: i for i, word in enumerate(vocab)}
idx2word = {i: word for word, i in word2idx.items()}

embedding_size = 10
window_size = 7
learning_rate = 0.01
epochs = 3000

W1 = np.random.rand(len(vocab), embedding_size)
W2 = np.random.rand(embedding_size, len(vocab))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x/e_x.sum(axis=0)

def one_hot_vector(word):
    vec = np.zeros(len(vocab))
    vec[word2idx[word]] = 1
    return vec

def get_training_data():
    training_data = []
    words = [word for sentence in tokenized_corpus for word in sentence]
    for i in range(window_size, len(words) - window_size):
        context = [words[j] for j in range(i - window_size, i + window_size + 1) if j != i]
        target = words[i]
        training_data.append((context, target))
    return training_data


def forward_pass(context):
    x = np.mean([one_hot_vector(k) for k in context], axis =0)
    h = np.dot(W1.T, x)
    u = np.dot(W2.T, h)
    y_pred = softmax(u)
    return y_pred, h, x

def back_propagation(y_pred,h, x, target):
    global W1, W2
    e = y_pred - one_hot_vector(target)
    dW2 = np.outer(h,e)
    dW1 = np.outer(x, np.dot(W2, e))
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2

training_data = get_training_data()
for epoch in range(epochs):
    loss=0 
    for context, target in training_data:
        y_pred, h, x = forward_pass(context)
        back_propagation(y_pred, h, x, target)
        loss -= np.log(y_pred[word2idx[target]] + 1e-9)
    if epoch % 100 == 0:
        print(f"epoch: {epoch}, Loss: {loss}")

embeddings = {word: W1[word2idx[word]] for word in vocab}
cbow_embeddings = pd.DataFrame.from_dict(embeddings, orient='index', columns=[f"Dim_{i+1}" for i in range(embedding_size)])
print(cbow_embeddings.head())