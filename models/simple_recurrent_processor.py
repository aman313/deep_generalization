import torch
import torch.nn as nn
from data_utils import data_gen_utils as gen
import torch.optim as optim
from data_utils import data_read_utils as read
from torch.autograd import Variable

class SequenceToNumberEncoder(nn.Module):
    '''
        Takes in a one hot encoded sequence and predicts the number it represents 
    '''
    
    def __init__(self):
        super(SequenceToNumberEncoder, self).__init__()
        self.lstm = nn.LSTM(10,1)
        self.linear1 = nn.Linear(1,20)
        self.linear2 = nn.Linear(20,1,False)
    
    
    def forward(self, input):
        lstm_out,lstm_cell = self.lstm(input,(Variable(torch.zeros(input.size(0),1),requires_grad=False),Variable(torch.zeros(input.size(0),1),requires_grad=False) ) )
        linear1_out = self.linear1(lstm_out[:,-1])
        linear2_out = self.linear2(linear1_out)
        return linear2_out
    

class RelativeDifferenceLoss(nn.Module):
    def __init__(self):
        super(RelativeDifferenceLoss,self).__init__()
    
    def forward(self, x,y):
        abs_diff_loss = nn.L1Loss()
        abs_diff = abs_diff_loss(x,y)
        return abs_diff/y if y.data.equal(torch.zeros(1)) else abs_diff
        

def run_epoch(net,train_data_gen,criterion,opt):
    net.train()
    for (X,y) in train_data_gen():
        X,y = Variable(X),Variable(y)
        opt.zero_grad()
        output = net(X)
        #print (output,y)
        loss = criterion(output,y)
        loss.backward()
        opt.step()

def test(net,test_data_gen,criterion,verbose=False):
    net.eval()
    total_loss = 0
    num_samples = 0
    for X,y in test_data_gen():
        X,y = Variable(X),Variable(y)
        num_samples += len(y)
        output = net(X)
        losses = criterion(output, y)
        if verbose:
            for x,p,g in zip(X.data.tolist(),output.data.tolist(),y.data.tolist()):
                print (x,p,g)
        total_loss += (criterion(output, y).sum())

    return total_loss/num_samples

def train(net,train_data_gen,test_data_gen,criterion,opt,num_epochs):
    for i in range(num_epochs):
        run_epoch(net, train_data_gen, criterion, opt)
        print ('Test loss after epoch ' , i, ' : ',test(net, test_data_gen,criterion) )
    
    print ('Final train loss  after epoch ' , i, ' : ',test(net, train_data_gen,criterion) )

encoder = read.one_hot_transformer([str(x) for x in range(10)])
train_file = '/Users/aman313/Documents/data/synthetic/pos_int_regression_train.csv'
test_file = '/Users/aman313/Documents/data/synthetic/pos_int_regression_test.csv'
batched_data_generator = read.batched_data_generator_from_file_with_replacement
criterion = RelativeDifferenceLoss()

'''
net = SequenceToNumberEncoder()
opt = optim.Adam(net.parameters(), lr=1e-3)
train(net,batched_data_generator(train_file, 100, 100,encoder),batched_data_generator(test_file,100,2,encoder),criterion,opt,100)

torch.save(net, 'model.pkl')
'''
net = torch.load('model.pkl')
test(net,batched_data_generator(test_file,100,1,encoder),criterion,True)
#print(torch.stack([encoder('3',1)]))
#print(torch.stack([encoder('33',2)]))

#print (net(Variable(torch.stack([encoder('3',1)]))))
#print (net(Variable(torch.stack([encoder('33',2)]))))
#print (net(Variable(torch.stack([encoder('333',3)]))))
#print (net(Variable(torch.stack([encoder('3333',4)]))))


        