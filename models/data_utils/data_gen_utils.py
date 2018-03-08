import random
from itertools import permutations
from itertools import combinations
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


def create_positive_integers_dataset(file_name,generator_funtion,filter_function,max_len=10,train_ratio=0.8,val_ratio=0.2,gen_args={}):
    with open(file_name+'train.csv','w') as out_file_train,open(file_name+'test.csv','w') as out_file_test,open(file_name+'val.csv','w') as out_file_val:
        out_file_train.write('X,y\n')
        out_file_test.write('X,y\n')
        out_file_val.write('X,y\n')
        count =0
        for X,y in filter( lambda x: filter_function(x[0]),generator_funtion(**gen_args)):
            count+=1
            X_y = X +','+ str(y)
            rnd = random.random()
            if rnd < val_ratio:
                out_file_val.write(X_y +'\n')
            elif rnd>=val_ratio and rnd <train_ratio: 
                out_file_train.write(X_y +'\n')
            else:
                out_file_test.write(X_y +'\n')
            
        print (count)
if __name__=='__main__':
    create_positive_integers_dataset(file_name='../../../data/synthetic/pos_int_regression_ml4_even_', max_len=5,generator_funtion=lambda :((str(x),x) for x in range(10000)),filter_function=lambda x:int(x)%2==0 )
    #for s in generate_all_possible_till_length(range(10),4):
     #   print(s,int(s))