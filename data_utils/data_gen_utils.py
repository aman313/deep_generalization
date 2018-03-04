import random

def generate_random_string_from_vocab(vocab,str_len):
    rs = ''
    for i in range(str_len):
        rs+=random.choice(vocab)
    return rs


def generate_random_strings_from_vocab(vocab,num_strings,max_str_len):
    allowed_str_lens = range(1,max_str_len)
    for i in range(num_strings):
        str_len = random.choice(allowed_str_lens)
        yield generate_random_string_from_vocab(vocab, str_len)

def generate_random_positive_integers(num_integers, max_len):
    vocab = [str(x) for x in range(10)]
    return generate_random_strings_from_vocab(vocab, num_integers, max_len)


def generate_positive_integers_dataset(size= 10000,max_len=10):
    for X in generate_random_positive_integers(size, max_len):
        yield (X, int(X))


def create_positive_integers_dataset(file_name,size= 10000,max_len=10):
    with open(file_name,'w') as out_file:
        out_file.write('X,y\n')
        for X,y in generate_positive_integers_dataset(size, max_len):
            X_y = X +','+ str(y)
            out_file.write(X_y +'\n')
            

if __name__=='__main__':
    create_positive_integers_dataset('/Users/aman313/Documents/data/synthetic/pos_int_regression_train.csv', 1000, 3)