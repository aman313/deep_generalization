import pandas as pd
import torch
import numpy as np
import ast

GPU=False

def one_hot_transformer(vocab):
    vocab_index = {elem:index for index,elem in enumerate(vocab)}
    def trans(str,max_len):
        one_hot = torch.zeros(max_len,len(vocab))
        for i in range(len(str)):
            char = str[i]
            try:
                one_hot[i,vocab_index[char]] = 1
            except KeyError:
                print('element not in vocab ', char)
                raise Exception
        return one_hot
    
    return trans

def batched_data_generator_from_file_with_replacement(file_name,batch_size,num_batches,transformer,data_type=np.int64):
    data = pd.read_csv(file_name,dtype={'X': data_type, 'y': data_type})
    print('Read file ',file_name)
    def generate_batches():
        for i in range(num_batches):
            batch_data = data.sample(n = batch_size,replace=True)
            X = batch_data.X.tolist()
            y = batch_data.y.tolist()
            #print(X[0])
            X,y = zip(*sorted(zip(X,y),key=lambda x:len(str(x[0])),reverse=True))
            seq_lens = [len(str(x)) for x in X]
            max_len = max(seq_lens)
            if GPU:
                yield ( [transformer(str(x),max_len) for x in X],torch.cuda.FloatTensor(y) )
            else:
                yield ( [transformer(str(x),max_len) for x in X],torch.FloatTensor(y) )
    return generate_batches
def batched_data_generator_from_file_with_replacement_for_string_to_seq_of_tuples(file_name,batch_size,num_batches,transformer):
    data = pd.read_csv(file_name,dtype={'X': np.int64, 'y': np.str})
    def generate_batches():
        for i in range(num_batches):
            batch_data = data.sample(n = batch_size,replace=True)
            X = batch_data.X.tolist()
            ystr = batch_data.y.tolist()
            y=[ast.literal_eval(k) for k in ystr]
            X,y = zip(*sorted(zip(X,y),key=lambda x:len(str(x[0])),reverse=True))
            seq_lens = [len(str(x)) for x in X]
            max_len = max(seq_lens)
#             print (y)
            yield ( [transformer(str(x),max_len) for x in X],torch.FloatTensor(y) )
    return generate_batches