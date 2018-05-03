from preprocessing import *
from model_classification import *

def get_wrong_imgs(net_, X, y, mode='train'):
    '''
    Function searches for badly recognized images.
    It stupidly outputs all images that were labelled in a wrong way.
    '''    
    net_.eval()
  
    predictions_from_model = net_(Variable(torch.FloatTensor(X)).cuda()).data.cpu().numpy()
    predictions = predictions_from_model.copy()
    predictions[predictions_from_model<=0.5] = 0
    predictions[predictions_from_model>0.5] = 1
    arr = np.concatenate((predictions[:,None], y[:,None], np.arange(len(X))[:,None]), axis=1).astype(int)
    mask = arr[:,0]!=arr[:,1]
    arr = arr[mask,...]
    idx_wrong_imgs = arr[:,2]
    wrong_imgs, true_labels = swap_back(X[idx_wrong_imgs]), y[idx_wrong_imgs]
    preds = np.array(predictions_from_model[idx_wrong_imgs])
    
    N = wrong_imgs.shape[0]
    print('number of badly recognized images:', N)
    print('({} mode)'.format(mode))
    if N > 15:
        N = 10
    n_rows = N//5+1
    fig, ax = plt.subplots(nrows=n_rows, ncols=5, figsize=(20,5*n_rows))
    for i in range(n_rows):
        for j in range(5):
            try:
                idx = i*5+j
                ax[i,j].imshow(wrong_imgs[idx])
                ax[i,j].set_title('prediction: {:.2f} | truth: {}'.format(preds[idx], true_labels[idx]))
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])  
            except IndexError:
                pass
    plt.show()
            
def get_strange_imgs_by_inner(model, X, tol=0.1):
    '''
    Function calculates images lying between classes.
    It takes inner representations of input images (after first dense layer) and searches for farthest ones from classes' centers.
    '''
    X_clocks, X_crocs = swap(get_clocks()), swap(get_crocs())
    inner_clock, inner_croc = model.get_latent(Variable(torch.FloatTensor(X_clocks)).cuda()),\
                          model.get_latent(Variable(torch.FloatTensor(X_crocs)).cuda())
    aver_clock, aver_croc = inner_clock.mean(0).data.cpu().numpy()[None,:], \
                            inner_croc.mean(0).data.cpu().numpy()[None,:]
    inner_all = model.get_latent(Variable(torch.FloatTensor(swap(X))).cuda()).data.cpu().numpy()
    
    dist_from_clock = np.linalg.norm(inner_all-aver_clock, axis=1)[:,None]
    dist_from_croc = np.linalg.norm(inner_all-aver_croc, axis=1)[:,None]
    distances = np.concatenate((dist_from_clock, dist_from_croc, np.arange(len(X))[:,None]),axis=1)
    
    mask = np.abs(distances[:,0]-distances[:,1]) < tol
    distances = distances[mask,...]
    idx_strange_imgs = distances[:,2].astype(int)
    strange_imgs = X[idx_strange_imgs]    
    
    N = strange_imgs.shape[0]
    n_rows = N//5+1
    fig, ax = plt.subplots(nrows=n_rows, ncols=5, figsize=(20,5*n_rows))
    for i in range(5):
        for j in range(5):
            try:
                ax[i,j].imshow(strange_imgs[i*5+j])
                ax[i,j].set_xticks([])
                ax[i,j].set_yticks([])  
            except IndexError:
                pass
    plt.show()
