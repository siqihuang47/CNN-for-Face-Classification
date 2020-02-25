#!/usr/bin/env python
# coding: utf-8

# In[1]:


# from baseline_copy import *
from baseline_cnn import *
import time


# In[2]:


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# In[3]:


# Check if your system supports CUDA
use_cuda = torch.cuda.is_available()

# Setup GPU optimization if CUDA is supported
if use_cuda:
    computing_device = torch.device("cuda:0")
    extras = {"num_workers": 1, "pin_memory": True}
    print("CUDA is supported")
else: # Otherwise, train on the CPU
    computing_device = torch.device("cpu")
    extras = False
    print("CUDA NOT supported")

    
net=Nnet()
    
# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     net = nn.DataParallel(net)

    
net.to(computing_device)
net.apply(weights_init)

# Print the model
print(net)

# config_path = 'model_config/' + time.strftime("%I_%M%p", time.localtime()) + '.txt'
# f = open(config_path, "w")
# f.write(str(net))
# f.close()

#loss criteria are defined in the torch.nn package
criterion = nn.CrossEntropyLoss()

#Instantiate the gradient descent optimizer - use Adam optimizer with default parameters
optimizer = optim.Adam(net.parameters(),lr = 0.001)


# In[4]:

stop = False

transform = transforms.Compose([transforms.Resize(128), transforms.CenterCrop(128), transforms.ToTensor()])
dataset = loader('train.csv','/datasets/cs154-fa19-public/',transform=transform)
batch_size = 64
validation_split = .5
shuffle_dataset = True
random_seed= 42

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    

val_indices, train_indices= indices[split:], indices[:split]
print(split)
print("train indices: ")
# print(train_indices)
print()
print("vlaid indices: ")
# print(val_indices)

# break


# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)

# test_indices = list(range(256))
# tst_inds = [:]
# test_sampler = SubsetRandomSampler(tst_inds)




# In[8]:


import copy
# Track the loss across training
total_loss = []
total_accu = []

train_losses = []
train_accuracies = []

avg_minibatch_loss = []
avg_minibatch_accu = []
N = 50

all_models = []

valid_losses = []
valid_accuracies = []

xnloss = []

start = time.monotonic()

#num of epochs
n_epochs = 15

#Helper function for calculating accuracy
def calculate_accu(outputs, labels, batch_size):
    num_correct = torch.sum(torch.max(outputs, dim = 1)[1] == labels).item()
    return num_correct / batch_size

for epoch in range(n_epochs):
    
    epoch_start = time.monotonic()
    N_minibatch_loss = 0.0
    net.train()
    
    N_minibatch_accu = 0.0

    # Get the next minibatch of images, labels for training
    for minibatch_count, (images, labels) in enumerate(train_loader, 0):
        print("mini_batch", minibatch_count)
        # Zero out the stored gradient (buffer) from the previous iteration
        optimizer.zero_grad()
        # Put the minibatch data in CUDA Tensors and run on the GPU if supported
        images, labels = images.to(computing_device), labels.to(computing_device)
        # Perform the forward pass through the network and compute the loss
        outputs = net(images)
        
        loss = criterion(outputs, labels)
        accu = calculate_accu(outputs, labels, batch_size)
        # Automagically compute the gradients and backpropagate the loss through the network
        loss.backward()

        # Update the weights
        optimizer.step()    
        # Add this iteration's loss to the total_loss
        total_loss.append(loss.item())
        total_accu.append(accu)
        
        N_minibatch_loss += loss #N_minibatch_loss += loss.item()
        N_minibatch_accu += accu
        
        if minibatch_count % N == 49:
            #Print the loss averaged over the last N mini-batches
            N_minibatch_loss /= N
            N_minibatch_accu /= N
            
            print('Epoch %d, average minibatch %d loss: %.3f' % (epoch + 1, minibatch_count+1, N_minibatch_loss))
            # Add the averaged loss over N minibatches and reset the counter
            avg_minibatch_loss.append(N_minibatch_loss)
            avg_minibatch_accu.append(N_minibatch_accu)
            
            N_minibatch_loss = 0.0
            N_minibatch_accu = 0.0
    
    train_losses.append(np.average(total_loss))
    train_accuracies.append(np.average(total_accu))
    
    print("Finished", epoch + 1, "epochs of training")
    # TODO: Implement validation #with torch.no_grad():
    
    epoch_RT = time.monotonic() - epoch_start
    formattedTime = time.strftime("%H:%M:%S", time.gmtime(epoch_RT))
    print('RUNTIME FOR EPOCH: ', epoch + 1, formattedTime)
    
   
    print("now doing validation performance")
    
    val_start = time.monotonic()
    net.eval()
    
    # TODO: Implement validation 
    with torch.no_grad():
        
        valid_loss = 0
        valid_accu = 0
        
        for minibatch_count, (images, labels) in enumerate(validation_loader, 0):
            
            #Apply current model to the data
            images, labels = images.to(computing_device), labels.to(computing_device)
            outputs = net(images)
            
            valid_accu += calculate_accu(outputs, labels, batch_size)
            valid_loss += criterion(outputs, labels).item()
        
        avg_valid_accu = valid_accu/minibatch_count
        avg_valid_loss = valid_loss/minibatch_count
        
        val_RT = time.monotonic() - val_start
        formattedTime = time.strftime("%H:%M:%S", time.gmtime(val_RT))
        print('RUNTIME OF VAL PERFORMANCE: ', formattedTime)   
        
        print("Valid loss for validation set is ", avg_valid_loss, "%.")
        print("Accuracy for validation set is", avg_valid_accu * 100, "%.")
    
#         xnloss.append(avg_valid_loss)

            
        if len(xnloss) < 2:
            xnloss.append(avg_valid_loss)
        elif xnloss[-2] < xnloss[-1] < avg_valid_loss:
            xnloss.append(avg_valid_loss)
            if stop:
                print("EARLY STOPPED AT EPOCH: ", epoch + 1)
                break
        else:
            xnloss.append(avg_valid_loss)
            
        valid_accuracies.append(avg_valid_accu)
    
    duration = time.monotonic() - start
    formattedTime = time.strftime("%H:%M:%S", time.gmtime(duration))
    print('TOTAL RUNTIME SO FAR :) ', formattedTime)
    
    #saving model
    path = 'saved_models/baseline_fold2_' + time.strftime("%b%d/", time.localtime())
    os.makedirs(path, 0o777, exist_ok=True)
    savepath = path  + time.strftime("%I_%M%p", time.localtime()) + '.pt'
    torch.save(net, savepath)
    
finalTime = time.monotonic() - start
formattedTime = time.strftime("%H:%M:%S", time.gmtime(finalTime))
print('Total time with ESing: ', formattedTime)

# path = 'saved_models/' + time.strftime("%b%d/", time.localtime())
# os.makedirs(path, 0o777, exist_ok=True)
# lastmodel = path  + 'lastModel.pt'
torch.save(net, 'baseline_fold2.pt')


import matplotlib.pyplot as plt


# ### Training and Validation Loss Curves

# In[ ]:


graph_path = 'graphs/' + time.strftime("%I_%M%p", time.localtime()) + "_baseline_fold2" + '.png'
# os.makedirs(path, 0o777, exist_ok=True)
# graph_savpth = graph_path  + time.strftime("%I_%M%p", time.localtime()) + '.png'


graph_title = 'Training vs Validation Losses for Baseline fold-2 for 15 epochs'

fig = plt.figure(figsize=(10, 8))
ax = plt.subplot()
ax.plot(np.arange(1, len(train_losses) + 1), train_losses, label='Training Losses')
ax.plot(np.arange(1, len(xnloss) + 1), xnloss, label='Validation Losses')
ax.set(xlabel='Number of Epochs', ylabel='Losses',
           title=graph_title)
leg = ax.legend() #loc=4)
fig.savefig(graph_path)
# plt.show()

config_path = 'model_config/' + time.strftime("%I_%M%p", time.localtime()) + "_baseline_fold2" + '.txt'
f = open(config_path, "w")
f.write(str(net))
f.close()


### Plot on Training and Testing Accuracy Curves

# In[ ]:


# graph_title = 'Training vs Validation Accuracies for for Baseline CNN with Early Stopping'

# fig = plt.figure(figsize=(10, 8))
# ax = plt.subplot()
# ax.plot(np.arange(1, len(train_accuracies) + 1), np.array(train_accuracies) * 100, label='Training Accuracies(%)')
# ax.plot(np.arange(1, len(valid_accuracies) + 1), np.array(valid_accuracies) * 100, label='Validation Accuracies(%)')
# ax.set(xlabel='Number of Epochs', ylabel='Accuracy(%)',
#            title=graph_title)
# leg = ax.legend(loc=4)
# fig.savefig('graphs/baseline_train_test_accu.png')
# plt.show()

# In[ ]:


# print('The test accuracy is ' + str(test_accu/minibatch_count))


# ### Filter Maps

# In[ ]:




