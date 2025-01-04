import torch
import torch.nn.functional as F
import string

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
            
def generate_words(rng, num_classes, layer, itos, bigram_itos, bigram_stoi):
    g = torch.Generator().manual_seed(2147483647)
    out = []
    for i in range(rng):
        word = ['aa']
        ix = bigram_stoi['aa']
        while True:
            input_enc = one_hot(torch.tensor([ix]), num_classes = num_classes)
            p = layer.call(input_enc)
            next = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            nextChar = itos[next]
            word.append(nextChar)
            if nextChar == '.':
                break
            currentBigram = bigram_itos[ix]
            ix = bigram_stoi[''.join([currentBigram[1:], nextChar])]
        out.append(''.join(word))
    return out

def calculate_loss(y_pred, y_train, size):
    return -y_pred[torch.arange(size), y_train].log().mean()
    
def one_hot(xs, num_classes):
    return F.one_hot(xs, num_classes = num_classes).float()

def create_train_data(words, bigram_stoi, stoi):
    xs = []
    ys = []
    for word in words:
        word = '.' + word + '.'
        for i in range(len(word) - 2):
            bigram = ''.join([word[i], word[i+1]])
            next = word[i+2]
            xs.append(bigram_stoi[bigram])
            ys.append(stoi[next])
    return torch.tensor(xs), torch.tensor(ys)

def make_bigram_mappings(words):
    chars = []
    
    letters = string.ascii_lowercase
    start_dots = ['.' + letter for letter in letters]
    middle_combinations = [a + b for a in letters for b in letters]
    end_dots = [letter + '.' for letter in letters]
    chars = start_dots + middle_combinations + end_dots
    chars = sorted(list(set(chars)))
    itos = {index + 1: value for index, value in enumerate(chars)}
    stoi = {value: index + 1 for index, value in enumerate(chars)}
    itos[0] = ".."
    stoi[".."] = 0
    return (itos, stoi)
    
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
    bigram_itos, bigram_stoi = make_bigram_mappings(names)
    xs, ys = create_train_data(names, bigram_stoi, stoi)
    xs_encoded = one_hot(xs, num_classes = len(bigram_stoi))
    # 10x729
    # 729 weights x 27 neurons
    # dot product is: 10x27 = 10 outputs with probability of 1 of 27 chars for each output
    layer = Layer(n_inputs = len(bigram_stoi), n_neurons = 27, g = g)
    layer.train(xs_encoded, ys, xs.nelement(), 10000)
    out = generate_words(50, len(bigram_stoi), layer, itos, bigram_itos, bigram_stoi)
    
    for word in out:
        print(word)

#2.068171262741089
#2.0681703090667725
#2.068169593811035
#2.068169355392456
#2.0681684017181396
#2.0681676864624023
#2.068166971206665
#2.068166494369507
#aadexza.
#aalealius.
#aarleikaydnevonimittain.
#aallayk.
#aaka.
#aada.
#aarleigha.
#aalton.
#aamilias.
#aamoriellavo.
#aan.
#aarteda.
#aakaley.
#aam.
#aaside.
#aa.
#aankaviyah.
#aantlspihiliven.
#aatahlas.
#aa.
#aansord.
#aaleenlen.
#aann.
#aaisan.
#aa.
#aarridynne.
#aazer.
#aader.
#aa.
#aaniyat.
#aarcielle.
#aanabelarl.
#aadiya.
#aaqee.
#aanyanni.
#aanah.
#aarien.
#aan.
#aavi.
#aasill.
#aadavaysor.
#aamyson.
#aarsie.
#aanix.
#aaley.
#aariseriyen.
#aan.
#aaille.
#aanahmie.
#aan.
