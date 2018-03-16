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
# import matplotlib.pyplot as plt
import time
import datetime
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence
from torch.nn.utils.rnn import pad_packed_sequence
from indice_and_multiplier_subnetwork import *
import sys
use_preTrained=True
use_LSTM=True
class SequenceToNumberEncoderCompositional(nn.Module):
    '''
        Takes in a one hot encoded sequence and predicts the number it represents 
    '''
    
    def __init__(self):
        super(SequenceToNumberEncoderCompositional, self).__init__()
        print("Creating network use pretrain",use_preTrained)
#         self.digitScalarNet=SequenceToScalarAndMultiplierPredictor()
        if use_preTrained:
            self.preTrainedNets=nn.ModuleList()
            self.digitScalarNet = torch.load('model_upper_50.pkl')
            for param in self.digitScalarNet.parameters():
                param.requires_grad = False
            self.preTrainedNets.append(self.digitScalarNet)
        if(use_preTrained):
            if(use_LSTM):
                self.rnn1 = nn.LSTM(10,19,batch_first=True,num_layers=1)
            else:
                self.rnn1 = nn.GRU(10,19,batch_first=True,num_layers=1)
        else:
            if(use_LSTM):
                self.rnn1 = nn.LSTM(10,20,batch_first=True,num_layers=1)
            else:
                self.rnn1 = nn.GRU(10,20,batch_first=True,num_layers=1)
        if(use_LSTM): 
            self.rnn2 = nn.LSTM(20,20,batch_first=True,num_layers=1)
        else:
            self.rnn2 = nn.GRU(20,20,batch_first=True,num_layers=1)
        self.linear1 = nn.Linear(20,1)

        
    
    def get_stacked_last_slices(self,unpacked,seq_lens):
        last_slices = []
        for i in range(unpacked.size(0)):
            last_slices.append(unpacked[i,seq_lens[i]-1,:])
        return torch.stack(last_slices)
    
    def forward(self, input):
        if isinstance(input,PackedSequence):
            if(use_preTrained):
                if(use_LSTM): 
                    rnn1_out,_ = self.rnn1(input,(Variable(torch.zeros(1,input.batch_sizes[0],19),requires_grad=False),Variable(torch.zeros(1,input.batch_sizes[0],19),requires_grad=False) ) )
                else:
                    rnn1_out,_ = self.rnn1(input,Variable(torch.zeros(1,input.batch_sizes[0],19),requires_grad=False))
                rnn1_out_unpacked,seq1_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn1_out,batch_first=True)
                digit_scalar_out=self.digitScalarNet(input)
    #             print(rnn1_out_unpacked)
    #             print(digit_scalar_out)
                rnn2_in_unpacked=torch.cat([rnn1_out_unpacked,digit_scalar_out],2)
    #             print(rnn2_in_unpacked)
    #             print(len(seq1_lens),input.batch_sizes[0])
    #             raise("break")
                rnn2_in=torch.nn.utils.rnn.pack_padded_sequence(rnn2_in_unpacked,seq1_lens,batch_first=True)
            else:
                if(use_LSTM):
                    rnn1_out,_ = self.rnn1(input,(Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False),Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False) ) )
                else:
                    rnn1_out,_ = self.rnn1(input,Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False))
                rnn2_in=rnn1_out
            if(use_LSTM):
                rnn2_out,_=self.rnn2(rnn2_in,(Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False),Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False) ) )
            else:
                rnn2_out,_=self.rnn2(rnn2_in,Variable(torch.zeros(1,input.batch_sizes[0],20),requires_grad=False))
            rnn2_out_unpacked,seq2_lens = torch.nn.utils.rnn.pad_packed_sequence(rnn2_out,batch_first=True)
#             linear_in=rnn2_in_unpacked.view(input.batch_sizes[0],-1)
#             print(rnn2_out_unpacked)
            linear_in = self.get_stacked_last_slices(rnn2_out_unpacked, seq2_lens)
#             print(linear_in)
            
        else:
            raise("Unimplemented for non packed sequences")
        linear1_out = torch.nn.ReLU()(self.linear1(linear_in))
        #linear2_out = torch.nn.ReLU()(self.linear2(linear1_out))
#         print(linear1_out[0])
        return linear1_out
    

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
        X,y = stack_and_pack(X,[len(str(int(x))) for x in y.tolist()],True),Variable(y)
        opt.zero_grad()
        output = net(X)
#         print("output:-",output)
#         
#         
#         print("y:-",y)
        #print (output,y)
        loss = criterion(output,y)
        loss.backward()
#         [subModule.zero_grad() for subModule in net.preTrainedNets]
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
                    print(batch_x[i],[batch_y[i]])
                    
                    yield ([batch_x[i]],[batch_y[i]])
            
        return single_generator
        
    if verbose:
        generator = present_single(test_data_gen)
        
    for X,y in generator():
        
        X,y = stack_and_pack(X,[len(str(int(x))) for x in y],True),Variable(y)
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
class RelativeDifferenceLoss(nn.Module):
    def __init__(self):
        super(RelativeDifferenceLoss,self).__init__()
    
    def forward(self, x,y):
        abs_diff_loss = nn.L1Loss(reduce=False)
        abs_diff = abs_diff_loss(x,y)
        return sum([p/q if not q==0 else p for p,q in zip(abs_diff,y.data.tolist())])/len(y.data.tolist())

def train_with_early_stopping(model_out,net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=10,verbose=False):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1,
                                                           patience=max_epochs_without_improv / 10, verbose=True,
                                                           threshold=0.0001,
                                                           threshold_mode='rel', cooldown=0, min_lr=1e-10, eps=1e-08)


    for i in range(num_epochs):
        train_loss = run_epoch(net, train_data_gen, criterion, optimizer)
        print("Train loss for epoch ",i," was ",train_loss)
        val_loss = test(net, val_data_gen, criterion, False)
        train_losses_list.append(train_loss.data)
        val_losses_list.append(val_loss.data)
        scheduler.step(val_losses_list[i][0])

        if i > 0:
            if best_val_loss.data.tolist()[0] ==0.0:
                break
            if ((best_val_loss.data.tolist()[0]-val_loss.data.tolist()[0])/best_val_loss.data.tolist()[0]) > tolerance:
                val_loss_not_improved = 0
                torch.save(net, model_out)
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

def doRun(modelPrefix,dataPrefix,usePreTrained):
#     space = {
#             'first_lstm_layer':hyperopt.hp.choice('first_lstm_layer',[]),
#             'first_dense_layer_count':hyperopt.hp.choice('first_dense_layer_count',[40,20]),
#             'second_dense_layer':hyperopt.hp.choice('second_dense_layer',[])
#         }
    global use_preTrained
    use_preTrained=usePreTrained
    encoder = read.one_hot_transformer(vocab_pos_int)
    train_file = '../../data/synthetic/pos_int_regression_ml{}_train.csv'.format(dataPrefix)
    val_file = '../../data/synthetic/pos_int_regression_ml{}_val.csv'.format(dataPrefix)
    test_file = '../../data/synthetic/pos_int_regression_ml{}_test.csv'.format(dataPrefix)

    batched_data_generator = read.batched_data_generator_from_file_with_replacement
    criterion = RelativeDifferenceLoss()
# #
    date=datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    model_name='model_compositional_{}'.format(modelPrefix)+date+'.pkl'
    # model_name='model_compositional_4_first_even_no_pretrain_2018-03-16 01-04-17.pkl'
    print("Training:-",model_name)
    net = SequenceToNumberEncoderCompositional()
    # net = torch.load(model_name)
    opt = optim.Adam(net.parameters(), lr=1e-2)
    train_losses,val_losses =train_with_early_stopping(model_name,net,batched_data_generator(train_file, 10, 10000,encoder),batched_data_generator(val_file,100,30,encoder),criterion,opt,10000,max_epochs_without_improv=50,verbose=True)
#
# #      
#     
#     model_name='model_compositional_4_first_even_no_pretrain_2018-03-16 01-04-17.pkl'
    net = torch.load(model_name)
    print('Original size test loss')
#     print(test(net,batched_data_generator(test_file,200,2,encoder),criterion,True))
# #     print('New size test loss')
#     print(test(net,batched_data_generator(val_file,200,2,encoder),criterion,True))
#     plot_pred_gold(net, batched_data_generator(val_file,200,2,encoder), model_name+'_val.png')
#     plot_pred_gold(net, batched_data_generator(test_file,200,2,encoder), model_name+'_test.png')
#     '''
    
    #print(torch.stack([encoder('3',1)]))
    #print(torch.stack([encoder('33',2)]))
    
    #print (net(Variable(torch.stack([encoder('3',1)]))))
    #print (net(Variable(torch.stack([encoder('33',2)]))))
    #print (net(Variable(torch.stack([encoder('333',3)]))))
    #print (net(Variable(torch.stack([encoder('3333',4)]))))

if __name__ == '__main__':
    doRun('4_first_even_no_pretrain_','4_first_even',False)