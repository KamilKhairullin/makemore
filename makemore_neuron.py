import torch
import torch.nn.functional as F

WORD_DELIMITER = '.'

class Layer:
    
    def __init__(self, n_inputs, n_neurons, g):
        self.W = torch.randn((n_inputs, n_neurons), generator = g, requires_grad=True)

    def call(self, x):
        logits = x @ self.W
        counts = logits.exp()
        yp = counts / torch.sum(counts, 1, keepdim=True)
        return yp
         
    def train(self, x_train, y_train, num, epochs):
        for i in range(epochs):
            predictions = layer.call(x_train)
            loss = calculate_loss(predictions, y_train, num)
            print(loss.item())
            layer.W.grad = None
            loss.backward()
            layer.W.data += layer.W.grad * -50
            
def generate_words(num, layer):
    g = torch.Generator().manual_seed(2147483647)
    out = []
    for i in range(num):
        word = []
        ix = 0
        while True:
            input_enc = one_hot(torch.tensor([ix]), num_classes = 27)
            p = layer.call(input_enc)
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            word.append(itos[ix])
            if ix == 0:
                break
        out.append(''.join(word))
    return out

def calculate_loss(y_pred, y_train, size):
    return -y_pred[torch.arange(size), y_train].log().mean()
    
def one_hot(xs, num_classes):
    return F.one_hot(xs, num_classes = num_classes).float()

def create_train_data(words, itos, stoi):
    xs = []
    ys = []
    for word in words:
        word = '.' + word + '.'
        for ch1, ch2 in zip(word, word[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])
    return torch.tensor(xs), torch.tensor(ys)

def make_mappings(words):
    chars = sorted(list(set(''.join(words))))
    
    itos = {index+1: value for index, value in enumerate(chars)}
    stoi = {value: index+1 for index, value in enumerate(chars)}
    
    itos[0] = WORD_DELIMITER
    stoi[WORD_DELIMITER] = 0
    return (itos, stoi)

def load_dataset():
    names = open("names.txt", "r").read().splitlines()
    return names

if __name__ == '__main__':
    g = torch.Generator().manual_seed(2147483647)
    names = load_dataset()
    itos, stoi = make_mappings(names)
    xs, ys = create_train_data(names, itos, stoi)
    num = xs.nelement()
    xs_encoded = one_hot(xs, num_classes = 27)
    
    # 27 weights x 27 neurons
    layer = Layer(n_inputs = 27, n_neurons = 27, g = g)
    layer.train(xs_encoded, ys, num, 100)
    print(generate_words(10, layer))
