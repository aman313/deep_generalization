import random
from itertools import permutations
from itertools import combinations
import csv

START_MARKER = ''
END_MARKER=''
from functools import partial
vocab_pos_int = [str(x) for x in range(10)]
vocab_pos_int.extend([x for x in [START_MARKER,END_MARKER] if len(x)>0])

def generate_random_string_from_vocab(vocab,str_len):
    rs = START_MARKER
    for i in range(str_len):
        rs+=random.choice(vocab)
    return rs +END_MARKER



def generate_all_possible_till_length(vocab, max_len): 
    for perm in [permutations(vocab,l) for l in range(max_len)]:
        for x in perm:
            if len(x) > 0:
                string =''.join([str(x[i]) for i in range(len(x)) ])
                yield (string,int(string))


def generate_random_strings_from_vocab(vocab,num_strings,max_str_len):
    allowed_str_lens = range(1,max_str_len)
    for i in range(num_strings):
        str_len = random.choice(allowed_str_lens)
        yield generate_random_string_from_vocab(vocab, str_len)

def generate_random_positive_integers(num_integers, max_len):
    vocab = [str(x) for x in range(10)]
    return generate_random_strings_from_vocab(vocab, num_integers, max_len)

def generate_random_strings_from_vocab_of_length(vocab,num_strings,max_str_len):
    allowed_str_lens = range(max_str_len-1,max_str_len)
    for i in range(num_strings):
        str_len = random.choice(allowed_str_lens)
        yield generate_random_string_from_vocab(vocab, str_len)

def generate_random_positive_integers_of_length(num_integers, max_len):
    vocab = [str(x) for x in range(10)]
    return generate_random_strings_from_vocab_of_length(vocab, num_integers, max_len)


def generate_positive_integers_dataset(size= 10000,max_len=10):
    for X in generate_random_positive_integers(size, max_len):
        yield (X, int(X.replace(START_MARKER,'').replace(END_MARKER,'') ))

def strictly_ascending(string,reverse = False):
    vocab_index = {str(i):x for i,x in enumerate(vocab_pos_int)}
    str_indices = [vocab_index[x] for x in string]
    if len(str_indices) > 1 and len(set(str_indices)) == 1:
        return False
    if sorted(str_indices,reverse=reverse) == str_indices:
        return True
    print(sorted(str_indices,reverse=reverse) ,str_indices)
    return False
def generate_sequence_to_current_digit_and_multiplier_dataset(size= 10000,max_len=10):
    def generator():
        for X in generate_random_positive_integers_of_length(size, max_len):
            print(X)
            Y=[]
            for i in range(0,len(X)):
                #Y.append([int(X[i:i+1]),10**i])
                Y.append([float(X[i:i+1])/50])
            yield (X, Y)
    return generator
def generateCurrentDigit(decimal):
        
    return int(decimal[0])
        



def create_positive_integers_dataset(file_name,size= 10000,max_len=10,train_ratio=0.6,val_ratio=0.2):
    create_dataset(file_name,generate_positive_integers_dataset(size, max_len),train_ratio,val_ratio)
def create_sequence_to_current_digit_and_multiplier_dataset(file_name,size= 10000,max_len=10,train_ratio=0.8,val_ratio=0.2):
    create_dataset(file_name,generate_sequence_to_current_digit_and_multiplier_dataset(size, max_len),lambda _:True,train_ratio,val_ratio)
            

def create_dataset(file_name,generator_function,filter_function=lambda _:True,train_ratio=0.8,val_ratio=0.2):
    with open(file_name+'train.csv','w') as out_file_train,open(file_name+'test.csv','w') as out_file_test,open(file_name+'val.csv','w') as out_file_val:
        csv_train= csv.writer(out_file_train)
        csv_test= csv.writer(out_file_test)
        csv_val= csv.writer(out_file_val)
        csv_train.writerow(['X','y'])
        csv_test.writerow(['X','y'])
        csv_val.writerow(['X','y'])
        
        for X,y in filter(lambda x:filter_function(x[0]),generator_function()):
            #X_y = X +','+ str(y)
            rnd = random.random()
            if rnd < val_ratio:
                csv_val.writerow([X,y])
            elif rnd>=val_ratio and rnd <train_ratio: 
                csv_train.writerow([X,y])
            else:
                csv_test.writerow([X,y])
if __name__=='__main__':
    # create_positive_integers_dataset('/Users/arvind/Documents/data/synthetic/pos_int_regression_ml15_', 1000000, 15)
#     create_sequence_to_current_digit_and_multiplier_dataset('../../../data/synthetic/digit_and_multiplier_sequence_from_decimal_dataset5_', 10000, 5)
        create_dataset(file_name='../../../data/synthetic/pos_int_regression_ml4_first_even_',generator_function=lambda :((str(x),x) for x in range(10000)),filter_function=lambda x:int(x[0])%2==0 )
