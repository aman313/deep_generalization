import pandas as pd
import torch

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


def batched_data_generator_from_file_with_replacement(file_name,batch_size,num_batches,transformer):
    data = pd.read_csv(file_name)
    def generate_batches():
        for i in range(num_batches):
            batch_data = data.sample(n = batch_size)
            max_len = max([len(str(x)) for x in batch_data.X.tolist()])
            yield (torch.stack([transformer(str(int(x)),max_len) for x in batch_data.X.tolist()]),torch.FloatTensor(batch_data.y.tolist() ))
    return generate_batches
