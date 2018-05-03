from preprocessing import *
from model_classification import *

X,y = Preprocessing()
check_preprocessing(X, y)

# Let's take train:val:test ratio as 0.5 : 0.25 : 0.25 = 500 : 250 : 250
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.333, random_state=0)
X_train, X_val, X_test = swap(X_train), swap(X_val), swap(X_test)

#__________________________________

# Task 1

model = SimpleNet().cuda()

train_again=False
if train_again:    
    loss_history = training_process(model, X_train, y_train, X_val, y_val, n_epochs=100)
    torch.save(model.state_dict(), 'model_state')
else:
    model.load_state_dict(torch.load('model_state'))
    
print(compute_acc(model, X_train, y_train), compute_acc(model, X_val, y_val), compute_acc(model, X_test, y_test))

#___________________________________

# Task 2

from get_imgs import *

# 1st approach
get_wrong_imgs(model, X_train, y_train, 'train')
get_wrong_imgs(model, X_val, y_val, 'val')
get_wrong_imgs(model, X_test, y_test, 'test')

# 2nd approach
get_strange_imgs_by_inner(model, X, 0.2)

#___________________________________

# Task 3

from model_generation import *

ae = AE().cuda()
arr4train = get_crocs()

train_again=False
if train_again:    
    training_process_ae(ae, swap(arr4train), num_epochs=70)
    torch.save(ae.state_dict(), 'ae_state')
else:
    ae.load_state_dict(torch.load('ae_state'))
    
# One can check that AE performance works badly, reproduction capability is not enough to generate new well-recognized images.


