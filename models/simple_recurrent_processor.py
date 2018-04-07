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
import multiprocessing
import sys

GPU = read.GPU

class SequenceToNumberEncoder(nn.Module):
    '''
        Takes in a one hot encoded sequence and predicts the number it represents 
    '''
    
    def __init__(self):
        super(SequenceToNumberEncoder, self).__init__()
        self.num_layers = 4
        self.embedding = nn.Embedding(10, 1)
        self.lstm = nn.LSTM(1,20,batch_first=True,num_layers=self.num_layers)
        self.linear1 = nn.Linear(20,1)
            
    def get_stacked_last_slices(self,unpacked,seq_lens):
        last_slices = []
        for i in range(unpacked.size(0)):
            last_slices.append(unpacked[i,seq_lens[i]-1,:])
        return torch.stack(last_slices)
    
    def forward(self, input):
        if isinstance(input,tuple):
            input = PackedSequence(input[0],input[1])
            
        if isinstance(input, list):
            seq_lens = [len(x.tolist()) for x in input]
            input = [self.embedding(x) for x in input]
            input = stack_and_pack(input, seq_lens, True, True)
            
        if isinstance(input,PackedSequence):
            if GPU:
                lstm_out,_ = self.lstm(input,(Variable(torch.randn(self.num_layers,input.batch_sizes[0],20).cuda(),requires_grad=False),Variable(torch.randn(self.num_layers,input.batch_sizes[0],20).cuda(),requires_grad=False) ) )
            else:
                lstm_out,_ = self.lstm(input,(Variable(torch.randn(self.num_layers,input.batch_sizes[0],20),requires_grad=False),Variable(torch.randn(self.num_layers,input.batch_sizes[0],20),requires_grad=False) ) )

            lstm_out_unpacked,seq_lens = torch.nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
            linear_in = self.get_stacked_last_slices(lstm_out_unpacked, seq_lens)
        else:
            lstm_out,_ = self.lstm(input,(Variable(torch.randn(2,input.size(0),20).cuda(),requires_grad=False),Variable(torch.randn(2,input.size(0),20).cuda(),requires_grad=False) ) )
            linear_in = lstm_out[:,-1,:]
        linear1_out = self.linear1(linear_in)
        #linear2_out = torch.nn.ReLU()(self.linear2(linear1_out))
        del input
        return linear1_out
    
    
class RelativeDifferenceLoss(nn.Module):
    def __init__(self):
        super(RelativeDifferenceLoss,self).__init__()
    
    def forward(self, x,y):
        abs_diff_loss = nn.L1Loss(reduce=False)
        abs_diff = abs_diff_loss(x,y)
        rel_diff = [(p/q) if not q==0 else p for p,q in zip(abs_diff,y.data.tolist())]
        mean = sum(rel_diff)/len(y.data.tolist())
        var = sum([torch.abs(x-mean) for x in rel_diff ])/(len(y.data.tolist()))
        #loss_tensor = torch.stack([x.data for x in rel_diff])
        #print('Loss variance and mean',torch.var(loss_tensor), torch.mean(loss_tensor))
        #return Variable(torch.FloatTensor([torch.mean( loss_tensor) ]) ,requires_grad=True)#+ torch.var(loss_tensor)
        #mean = Variable(torch.FloatTensor([torch.mean( loss_tensor) ]),requires_grad=True)
        return mean 
        
def stack_and_pack(lst,seq_lens,pack=False,stack=True):
    if not stack:
        return lst
    if not pack:
        if GPU:
            return Variable(torch.stack(lst)).cuda()
        else:
            return Variable(torch.stack(lst))
    else:
        packed_cpu = pack_padded_sequence(Variable(torch.stack(lst)),seq_lens,True)
        #print('packed on cpu')
        if GPU:
            packed_gpu = PackedSequence(packed_cpu.data.cuda(),packed_cpu.batch_sizes)
            return packed_gpu
        #print('packed on gpu',packed_gpu.batch_sizes)
        #print(type(packed_gpu))
        return packed_cpu
        
def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    train_loss = 0
    num_batches = 0
    for (X,y) in train_data_gen():
        #print('generated batch')
        if GPU:
            X,y = stack_and_pack(X,[len(str(int(x))) for x in y.tolist()],False,False),Variable(y.cuda())
        else:
            X,y = stack_and_pack(X,[len(str(int(x))) for x in y.tolist()],False,False),Variable(y)

        opt.zero_grad()
        #print(type(X))
        output = net((X))
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
                    yield ([batch_x[i]],torch.FloatTensor([batch_y[i]]))
            
        return single_generator
        
    if verbose:
        generator = present_single(test_data_gen)
        
    for X,y in generator():
        if GPU:
            X,y = stack_and_pack(X,[len(str(int(x))) for x in y.tolist()],True),Variable(y.cuda())
        else:
            X,y = stack_and_pack(X,[len(str(int(x))) for x in y.tolist()],True),Variable(y)
        num_batches += 1
        output = net(X)
        avg_loss = criterion(output, y)
        if verbose:
            if isinstance(X, PackedSequence):
                x_list = X.data.data.tolist()
            else:
                x_list = X.data.tolist()
            print ('x,y,o,l',x_list,y.data.tolist(),output.data.tolist(),avg_loss.data.tolist())
        
        total_loss += (avg_loss)
    return total_loss/num_batches

import gc
def train_with_early_stopping(net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=20,verbose=False,model_out=''):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    scheduler =torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, 
                                                         patience=90, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
    for i in range(num_epochs):
        #print('start epoch ',i)
        train_loss = run_epoch(net, train_data_gen, criterion, optimizer)
        val_loss = test(net, val_data_gen, criterion, False)
        if GPU:
            train_losses_list.append(train_loss.data.cpu())
            val_losses_list.append(val_loss.data.cpu())
            del train_loss,val_loss

        else:
            train_losses_list.append(train_loss.data)
            val_losses_list.append(val_loss.data)
        scheduler.step(val_losses_list[i][0])
        #optimizer.step()
        if i > 0:
            if best_val_loss[0] ==0.0:
                break
            if ((best_val_loss[0] -val_losses_list[i][0])/best_val_loss[0]) > tolerance:
                val_loss_not_improved = 0
                torch.save(net, model_out)
            else:
                val_loss_not_improved +=1
        if verbose:
            if i%10 ==0:
                print ('Epoch',i)
                print ('Train loss',train_losses_list[i][0])
                print ('Val loss', val_losses_list[i][0])
                print('No improvement epochs ',val_loss_not_improved)
                sys.stdout.flush()
        if  best_val_loss is None or val_losses_list[i][0] < best_val_loss[0]:
            best_val_loss = val_losses_list[i]
        if val_loss_not_improved >= max_epochs_without_improv:
            print('Early stopping at epoch',i)
            break
        gc.collect()
    net = torch.load(model_out)
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


space = {
        'first_lstm_layer':hyperopt.hp.choice('first_lstm_layer',[]),
        'first_dense_layer_count':hyperopt.hp.choice('first_dense_layer_count',[40,20]),
        'second_dense_layer':hyperopt.hp.choice('second_dense_layer',[])
    }

#encoder = read.one_hot_transformer(vocab_pos_int)
encoder = lambda x,_:torch.LongTensor([int(a) for a in x])
train_file = '../../data/synthetic/pos_int_regression_ml4_first_odd_train.csv'
val_file = '../../data/synthetic/pos_int_regression_ml4_first_odd_val.csv'
test_file = '../../data/synthetic/pos_int_regression_ml4_first_odd_test.csv'
batched_data_generator = read.batched_data_generator_from_file_with_replacement
criterion = RelativeDifferenceLoss()
#criterion = nn.L1Loss()

def do_run(run_index,out_prefix):
    #sys.stdout = open(str(out_prefix)+str(run_index) + ".log", "w")

    net = SequenceToNumberEncoder()
    if GPU:
    	net = torch.nn.DataParallel(net)
    #net = torch.load('model.pkl')
    opt = optim.Adam(net.parameters(), lr=1e-3)
    #opt = optim.SGD()
    train_losses,val_losses =train_with_early_stopping(net,batched_data_generator(train_file, 10, 300,encoder),batched_data_generator(val_file,100,9,encoder),criterion,opt,5000,max_epochs_without_improv=100,verbose=True,model_out=out_prefix+str(run_index)+'.pkl')
    torch.save(net, out_prefix+str(run_index)+'.pkl')


def do_parallel_runs(num_par,out_prefix):
    jobs = []
    for i in range(num_par):
        p = multiprocessing.Process(target=do_run,args=(i,out_prefix))
        jobs.append(p)
        p.start()


#do_parallel_runs(1,'model_ml4_odd_')
do_run('','pos_int_regression_ml4_first_odd_embed')

'''
test_file_ml2 = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml4_first_odd_test.csv'
test_file_ml4 = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml4_first_even_test.csv'
net = torch.load('pos_int_regression_ml4_first_odd_1_stddev.pkl')
print('Original size test loss')
print(test(net,batched_data_generator(test_file_ml2,800,1,encoder),criterion,True))
print('New size test loss')
print(test(net,batched_data_generator(test_file_ml4,800,1,encoder),criterion,True))
plot_pred_gold(net, batched_data_generator(test_file_ml2,800,1,encoder), 'train_ml4__first_odd_test_ml4_first_odd_1_stddev.png')
plot_pred_gold(net, batched_data_generator(test_file_ml4,800,1,encoder), 'train_ml4_first_odd_test_ml4_first_even_1_stdev.png')
'''

#print(torch.stack([encoder('3',1)]))
#print(torch.stack([encoder('33',2)]))

#print (net(Variable(torch.stack([encoder('3',1)]))))
#print (net(Variable(torch.stack([encoder('33',2)]))))
#print (net(Variable(torch.stack([encoder('333',3)]))))
#print (net(Variable(torch.stack([encoder('3333',4)]))))


        
