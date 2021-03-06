import torch
import torch.nn as nn
import torch.optim as optim
from data_utils import data_read_utils as read
from data_utils.data_gen_utils import vocab_pos_int
from torch.autograd import Variable
import hyperopt
from hyperopt.fmin import fmin
from hyperopt import tpe
from hyperopt.mongoexp import MongoTrials
from hyperopt.base import Trials, STATUS_OK
import matplotlib.pyplot as plt
import time
import datetime
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
import sys
class SequenceToScalarAndMultiplierPredictor(nn.Module):
    '''
        Takes in a one hot encoded sequence and predicts the number it represents 
    '''
    
    def __init__(self):
        super(SequenceToScalarAndMultiplierPredictor, self).__init__()
        self.numlayers=1
        self.lstm = nn.LSTM(10,1,batch_first=True,num_layers=self.numlayers)
        
    
    def get_stacked_last_slices(self,unpacked,seq_lens):
        last_slices = []
        for i in range(unpacked.size(0)):
            last_slices.append(unpacked[i,seq_lens[i]-1,:])
        return torch.stack(last_slices)
    
    def forward(self, input):
        if isinstance(input,PackedSequence):
            lstm_out,_ = self.lstm(input,(Variable(torch.randn(self.numlayers,input.batch_sizes[0],1),requires_grad=False),Variable(torch.randn(self.numlayers,input.batch_sizes[0],1),requires_grad=False) ) )
            lstm_out_unpacked,seq_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
            return lstm_out_unpacked
#             linear_in = self.get_stacked_last_slices(lstm_out_unpacked, seq_lens)
        else:
            lstm_out,_ = self.lstm(input,(Variable(torch.randn(1,input.size(0),2),requires_grad=False),Variable(torch.randn(1,input.size(0),2),requires_grad=False) ) )
            return lstm_out
#             linear_in = lstm_out[:,-1,:]
        #linear2_out = torch.nn.ReLU()(self.linear2(linear1_out))
        



def stack_and_pack(lst,seq_lens,pack=False):
    if not pack:
        return Variable(torch.stack(lst))
    else:
        return pack_padded_sequence(Variable(torch.stack(lst)),seq_lens,True)
    
        
def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    train_loss = 0
    num_batches = 0
    for (X,y) in train_data_gen():
        X,y = stack_and_pack(X,[len(x) for x in y.tolist()],True),Variable(y)
        opt.zero_grad()
        output = net(X)
#         print("output:-",output)
#         
#         
#         print("y:-",y)
        #print (output,y)
        loss = criterion(output,y)
        loss.backward()
        train_loss += loss
        num_batches+=1
        opt.step()
    return train_loss/num_batches

def test(net,test_data_gen,criterion,verbose=False):
    net.eval()
    total_loss = 0
    num_batches = 0
    generator = test_data_gen
    def present_single(batched_generator):
        def single_generator():
            for batch_x,batch_y in batched_generator():
                for i in range(len(batch_x)):
                    print(batch_x[i],torch.unsqueeze(batch_y[i],0))
                    
                    yield ([batch_x[i]],torch.unsqueeze(batch_y[i],0))
            
        return single_generator
        
    if verbose:
        generator = present_single(test_data_gen)
        
    for X,y in generator():
        X,y = stack_and_pack(X,[len(x) for x in y.tolist()],True),Variable(y)
        num_batches += 1
        output = net(X)
        avg_loss = criterion(output, y)
        if verbose:
            if isinstance(X, PackedSequence):
                x_list = X.data.data.tolist()
            else:
                x_list = X.data.tolist()
            print ('x,y,o,l',x_list,"\n",y.data.tolist(),"\n",output.data.tolist(),"\n",avg_loss.data.tolist())
            print("__________________")
#             sys.stdin.readline()
        total_loss += (avg_loss)
    return total_loss/num_batches

def train_with_early_stopping(net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=20,verbose=False):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    for i in range(num_epochs):
        train_loss = run_epoch(net, train_data_gen, criterion, optimizer)
        print("Train loss for epoch ",i," was ",train_loss)
        val_loss = test(net, val_data_gen, criterion, False)
        train_losses_list.append(train_loss)
        val_losses_list.append(val_loss)
        if i > 0:
            if best_val_loss.data.tolist()[0] ==0.0:
                break
            if ((best_val_loss.data.tolist()[0]-val_loss.data.tolist()[0])/best_val_loss.data.tolist()[0]) > tolerance:
                val_loss_not_improved = 0
            else:
                val_loss_not_improved +=1
        if verbose:
            if i%10 ==0:
                print ('Epoch',i)
                print ('Train loss',train_loss)
                print ('Val loss', val_loss)
                print('No improvement epochs ',val_loss_not_improved)
        if  best_val_loss is None or val_loss.data.tolist()[0] < best_val_loss.data.tolist()[0]:
            best_val_loss = val_loss
        if val_loss_not_improved >= max_epochs_without_improv:
            print('Early stopping at epoch',i)
            break
        
    return (train_losses_list,val_losses_list)

def train(net,train_data_gen,test_data_gen,criterion,opt,num_epochs):
    for i in range(num_epochs):
        run_epoch(net, train_data_gen, criterion, opt)
        if i%10 ==0 or i == num_epochs-1:
            print ('Test loss after epoch ' , i, ' : ',test(net, test_data_gen,criterion,True) )
            print (' Train loss  after epoch ' , i, ' : ',test(net, batched_data_generator(train_file, 100, 1,encoder),criterion,True) )
            

def plot_loss(loss,file_name =None):
    if not file_name:
        file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S') + '.png'
    plt.plot(loss)
    plt.savefig(file_name)
    plt.close()

def plot_pred_gold(net,data_generator,file_name=None):
    if not file_name:
        file_name = datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d_%H:%M:%S') + '.png'
    X,y = next(data_generator())
    X = stack_and_pack(X, [len(str(int(x))) for x in y.tolist()], True)
    pred = net(X)
    plt.scatter(y.tolist(),pred.data.tolist())
    plt.savefig(file_name)
    plt.close()

def network_constructor(params):
    pass

def objective_from_network(network_constructor,data_files):
    def objective(params):
        net = network_constructor(params)
        opt = params['optimizer'](net.parameters())
        train_losses,val_losses = train_with_early_stopping(net,batched_data_generator(data_files['train'],params['batch_size'],params['num_train_batches'],encoder),batched_data_generator(data_files['val'],params['batch_size'],params['num_test_batches'],encoder),criterion,opt,params['num_epochs'])
        return {'loss': val_losses[-1],'status': STATUS_OK}
    return objective
            
def explore_hyperparams(objective_provider,dataset, **fminargs ):
    best_point = fmin(fn=objective_provider(dataset),**fminargs)
    return best_point

if __name__ == '__main__':
#     space = {
#             'first_lstm_layer':hyperopt.hp.choice('first_lstm_layer',[]),
#             'first_dense_layer_count':hyperopt.hp.choice('first_dense_layer_count',[40,20]),
#             'second_dense_layer':hyperopt.hp.choice('second_dense_layer',[])
#         }
#     
    encoder = read.one_hot_transformer(vocab_pos_int)
    train_file = '/Users/arvind/Documents/data/synthetic/digit_and_multiplier_sequence_from_decimal_dataset_train.csv'
    val_file = '/Users/arvind/Documents/data/synthetic/digit_and_multiplier_sequence_from_decimal_dataset_val.csv'
    test_file = '/Users/arvind/Documents/data/synthetic/digit_and_multiplier_sequence_from_decimal_dataset_test.csv'
    batched_data_generator = read.batched_data_generator_from_file_with_replacement_for_string_to_seq_of_tuples
    criterion = torch.nn.MSELoss()
# #      
#     net = SequenceToScalarAndMultiplierPredictor()
#     opt = optim.Adam(net.parameters(), lr=1e-3)
#     train_losses,val_losses =train_with_early_stopping(net,batched_data_generator(train_file, 100, 6000,encoder),batched_data_generator(val_file,1000,100,encoder),criterion,opt,10000,max_epochs_without_improv=2,verbose=True)
#     torch.save(net, 'model_upper_719.pkl')
# # #      
#     
#     
    net = torch.load('model_upper_50.pkl')
    print('Original size test loss')
    print(test(net,batched_data_generator(test_file,200,1,encoder),criterion,True))
#     print('New size test loss')
#     print(test(net,batched_data_generator(test_file_ml4,200,1,encoder),criterion,True))
#     plot_pred_gold(net, batched_data_generator(test_file_ml2,200,1,encoder), 'train_ml2_test_ml2.png')
#     plot_pred_gold(net, batched_data_generator(test_file_ml4,200,1,encoder), 'train_ml2_test_ml4.png')
#     '''
    
    #print(torch.stack([encoder('3',1)]))
    #print(torch.stack([encoder('33',2)]))
    
    #print (net(Variable(torch.stack([encoder('3',1)]))))
    #print (net(Variable(torch.stack([encoder('33',2)]))))
    #print (net(Variable(torch.stack([encoder('333',3)]))))
    #print (net(Variable(torch.stack([encoder('3333',4)]))))


        