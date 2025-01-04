import torch

WORD_DELIMITER = '.'

def calculate_loss(N, words):
    logSum = 0
    n = 0
    for w in words:
        chs = ['.'] + list(w) + ['.']
        for ch1, ch2 in zip(chs, chs[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            prob = N[ix1, ix2]
            logProb = torch.log(prob)
            logSum += logProb
            n += 1
    return -logSum / n

def generate_words(N, r, generator):
    out = []
    for i in range(r):
        word = []
        ix = 0
        while True:
            p = N[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            word.append(itos[ix])
            if ix == 0:
                break
        out.append(''.join(word))
    return out

def normalize_bigram_matrix(N):
    N = N.float()
    N /= torch.sum(N, 1, keepdim=True)
    return N

def create_bigram_matrix(words, itos, stoi):
    N = torch.zeros(27, 27, dtype = torch.int32)
    for word in words:
        word = '.' + word + '.'
        for ch1, ch2 in zip(word, word[1:]):
            i1 = stoi[ch1]
            i2 = stoi[ch2]
            N[i1, i2] += 1
    return N

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
    names = load_dataset()
    itos, stoi = make_mappings(names)
    N = create_bigram_matrix(names, itos, stoi)
    N = normalize_bigram_matrix(N)
    g = torch.Generator().manual_seed(2147483647)
    print(generate_words(N, 20, g))
    print(calculate_loss(N, names))
