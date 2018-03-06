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

class SequenceToNumberEncoder(nn.Module):
    '''
        Takes in a one hot encoded sequence and predicts the number it represents 
    '''
    
    def __init__(self):
        super(SequenceToNumberEncoder, self).__init__()
        self.lstm = nn.LSTM(10,20,batch_first=True,num_layers=2)
        self.linear1 = nn.Linear(20,1)
    
    def forward(self, input):
        lstm_out,lstm_cell = self.lstm(input,(Variable(torch.randn(2,input.size(0),20),requires_grad=False),Variable(torch.randn(2,input.size(0),20),requires_grad=False) ) )
        linear1_out = torch.nn.ReLU()(self.linear1(lstm_out[:,-1]))
        #linear2_out = torch.nn.ReLU()(self.linear2(linear1_out))
        return linear1_out
    
    
class RelativeDifferenceLoss(nn.Module):
    def __init__(self):
        super(RelativeDifferenceLoss,self).__init__()
    
    def forward(self, x,y):
        abs_diff_loss = nn.L1Loss(reduce=False)
        abs_diff = abs_diff_loss(x,y)
        return sum([p/q if not q==0 else p for p,q in zip(abs_diff,y.data.tolist())])/len(y.data.tolist())
        
def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    train_loss = 0
    num_batches = 0
    for (X,y) in train_data_gen():
        X,y = Variable(X),Variable(y)
        opt.zero_grad()
        output = net(X)
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
                for i in range(batch_x.size()[0]):
                    yield (torch.stack([batch_x[i]]),torch.FloatTensor([batch_y[i]]))
            
        return single_generator
        
    if verbose:
        generator = present_single(test_data_gen)
        
    for X,y in generator():
        X,y = Variable(X),Variable(y)
        num_batches += 1
        output = net(X)
        avg_loss = criterion(output, y)
        if verbose:
            print ('x,y,o,l',X.data.tolist(),y.data.tolist(),output.data.tolist(),avg_loss.data.tolist())
        
        total_loss += (avg_loss)
    return total_loss/num_batches

def train_with_early_stopping(net,train_data_gen,val_data_gen,criterion,optimizer,num_epochs,tolerance=0.001,max_epochs_without_improv=20,verbose=False):
    val_loss_not_improved=0
    best_val_loss = None
    train_losses_list = []
    val_losses_list = []
    for i in range(num_epochs):
        train_loss = run_epoch(net, train_data_gen, criterion, optimizer)
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
    pred = net(Variable(X))
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

encoder = read.one_hot_transformer(vocab_pos_int)
train_file = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml2_train.csv'
val_file = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml2_val.csv'
test_file = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml2_test.csv'
batched_data_generator = read.batched_data_generator_from_file_with_replacement
criterion = RelativeDifferenceLoss()

net = SequenceToNumberEncoder()
#net = torch.load('model.pkl')
opt = optim.Adam(net.parameters(), lr=1e-3)
train_losses,val_losses =train_with_early_stopping(net,batched_data_generator(train_file, 100, 100,encoder),batched_data_generator(val_file,150,1,encoder),criterion,opt,1000,max_epochs_without_improv=50,verbose=True)
torch.save(net, 'model_ml2.pkl')

'''
test_file_ml2 = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml2_test.csv'
test_file_ml4 = '/Users/aman313/Documents/data/synthetic/pos_int_regression_ml4_train.csv'

net = torch.load('model_ml2.pkl')
print('Original size test loss')
print(test(net,batched_data_generator(test_file_ml2,200,1,encoder),criterion,True))
print('New size test loss')
print(test(net,batched_data_generator(test_file_ml4,200,1,encoder),criterion,True))
plot_pred_gold(net, batched_data_generator(test_file_ml2,200,1,encoder), 'train_ml2_test_ml2.png')
plot_pred_gold(net, batched_data_generator(test_file_ml4,200,1,encoder), 'train_ml2_test_ml4.png')
'''

#print(torch.stack([encoder('3',1)]))
#print(torch.stack([encoder('33',2)]))

#print (net(Variable(torch.stack([encoder('3',1)]))))
#print (net(Variable(torch.stack([encoder('33',2)]))))
#print (net(Variable(torch.stack([encoder('333',3)]))))
#print (net(Variable(torch.stack([encoder('3333',4)]))))


        