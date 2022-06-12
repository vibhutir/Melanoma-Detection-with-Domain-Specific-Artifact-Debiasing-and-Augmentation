import os
import matplotlib.pyplot as plt
import time
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from globalbaz import args, DP, device
from dataset import *
from models import *
from test import *
from train_epoch_variations import *


# Setting seeds for reproducibility
def set_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# Function used to plot the curves for loss and accuracy:
def plot_curves(auc):
    # Plotting the AUC curve:
    plt.style.use('ggplot')
    plt.title('Validation AUC')
    plt.ylim(0, 1)
    plt.xlim(0, args.n_epochs)
    plt.xticks(range(0, args.n_epochs))
    plt.ylabel('AUC')
    plt.xlabel('epoch')
    # Plotting the test accuracy (red):
    plt.plot(auc, color='red', label='test')
    return plt
    print('Done!')


# Main training function
def run(df_train, transforms_train, transforms_val, transforms_marked,
        criterion, criterion_aux, criterion_aux2, fold=None):

    df_this = df_train
    # Setting different number of units for fully connected layer based on feature extractor output
    
    in_ch = 2048
    if args.META:
        n_meta_features = 583
        in_ch = in_ch + 128 # Output size of MetaNN is 128 and output size of Resnext is 2048
        

    # Loading training data
    dataset_train = SIIMISICDataset(df_this, 'train', 'train', transform=transforms_train, transform2=transforms_marked)
    
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
                                               sampler=RandomSampler(dataset_train),
                                               num_workers=args.num_workers, drop_last=True)

    # defining feature extractor model and sending to gpu
    model_encoder = ResNext101()
    model_classifier = ClassificationHead(out_dim=args.out_dim, in_ch=in_ch)  # Creating main classification head 
    
    model_encoder = model_encoder.to(device)  # Sending feature extractor to GPU
    model_classifier = model_classifier.to(device)  # Sending classifier head to GPU
    
    if args.META:
        meta_encoder = MetaNN(n_meta_features)
        meta_encoder = meta_encoder.to(device)

    if args.debias_config != 'baseline':
        if args.deep_aux:
            model_aux = AuxiliaryHead2(num_aux=args.num_aux, in_ch=2048)
        else:
            model_aux = AuxiliaryHead(num_aux=args.num_aux, in_ch=2048)  # defining auxiliary head

        model_aux = model_aux.to(device)  # sending auxiliary head to GPU
        
    # Defining main optimizer used accross all models
    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, (list(model_encoder.parameters()) + list(model_classifier.parameters()))),
        lr=args.lr_base, momentum=args.momentum)

    # defining auxiliary optimisers for TABE
    if args.debias_config == 'TABE':
        optimizer_confusion = optim.SGD(model_encoder.parameters(), lr=args.lr_class,
                                        momentum=args.momentum)  # Defining confusion optimiser (boosted encoder optimiser)
        optimizer_aux = optim.SGD(model_aux.parameters(), lr=args.lr_class, momentum=args.momentum)  # defining auxiliary classification optimiser
# ---------------------------------------------------------------------------------------

    # defining variables to save scores to
    auc_max = 0.
    val_losses = []
    auc_lst = [0]

    # looping through epochs to train
    for epoch in range(1, args.n_epochs + 1):
        print(time.ctime(), 'Epoch:', epoch)  # printing time and epoch number
        
        if args.META:
            if args.debias_config == 'baseline':
                train_loss = train_epoch_META(model_encoder, meta_encoder, model_classifier, train_loader, optimizer, criterion)
                
            if args.debias_config == 'TABE':
                train_loss, train_loss_aux = train_epoch_TABE_META(model_encoder,meta_encoder, model_classifier, model_aux, train_loader,
                                                              optimizer, optimizer_aux, optimizer_confusion, criterion,
                                                              criterion_aux)
            
        else:
            if args.debias_config == 'baseline':
                train_loss = train_epoch_baseline(model_encoder, model_classifier, train_loader, optimizer, criterion)
                
            if args.debias_config == 'TABE':
                train_loss, train_loss_aux = train_epoch_TABE(model_encoder, model_classifier, model_aux, train_loader,
                                                              optimizer, optimizer_aux, optimizer_confusion, criterion,
                                                              criterion_aux)
        
        
        if args.debias_config == 'baseline':
            content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                           f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                           f' train loss: {np.mean(train_loss):.5f}'

        if args.debias_config == 'TABE':
            content = time.ctime() + ' ' + f'Epoch {epoch},' \
                                           f' lr: {optimizer.param_groups[0]["lr"]:.7f},' \
                                           f' train loss: {np.mean(train_loss):.5f},' \
                                           f' train loss aux: {np.mean(train_loss_aux):.5f}'

       
        print(content)

        # writing metrics to text file
        with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'), 'a') as appender:
            appender.write(content + '\n')

        # saving model if training on full data
        if epoch % args.save_epoch == 0:
            torch.save(
                {'model_state_dict': model_encoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
                 'epoch': epoch}, os.path.join(args.model_dir,
                                               f'{args.test_no}/encoder_all_data_Test{args.test_no}.pth'))
            
            if args.META:
                torch.save(meta_encoder.state_dict(), 
                           os.path.join(args.model_dir,f'{args.test_no}/meta_all_data_Test{args.test_no}.pth'))
                
            torch.save(model_classifier.state_dict(),
                       os.path.join(args.model_dir, f'{args.test_no}/classifier_all_data_Test{args.test_no}.pth'))
            print('model saved')

    # plotting learning curves and saving for examination to decide optimal epoch
    val_curve = plot_curves(auc_lst)
    val_curve.savefig(f'{args.plot_dir}/{args.test_no}/val_curve_Test{args.test_no}.pdf')


def main():

    # writing arguments to text file
    with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'),
              'a') as appender:
        appender.write(str(args) + '\n')

    # Loading training, validation and test dataframes
    df_train, df_val,df_test_atlasD, df_test_atlasC, df_test_ASAN,\
        df_test_MClassD, df_test_MClassC, mel_idx = get_df()

    # Selecting test data based on experiment
    df_test_lst = [df_test_atlasD, df_test_atlasC, df_test_ASAN, df_test_MClassD, df_test_MClassC]

    criterion, criterion_aux, criterion_aux2 = criterion_func(df_train)

    transforms_marked, transforms_train, transforms_val = get_transforms()

    #    
    if not args.test_only:  # Skipping training if test_only
        run(df_train, transforms_train, transforms_val,
            transforms_marked, criterion, criterion_aux, criterion_aux2)
    roc_plt_lst = []  # list of tuples of metrics needed to plot ROC curves
    for index, df in enumerate(df_test_lst):
        fpr, tpr, a_u_c, sensitivity, specificity = test(index, df, mel_idx, transforms_val)
        roc_plt_lst.append((fpr, tpr, a_u_c, sensitivity, specificity))
        #saliency(index, df, transforms_val)  # Plotting saliency maps
    ROC_curve(roc_plt_lst)  # Plotting ROC curves
    # Pickling info needed to plot custom ROC plots with misc_code/ROC_plots.py
    with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}_roc_plt_lst.pkl'),
              'wb') as f:
        pickle.dump(roc_plt_lst, f)


if __name__ == '__main__':

    # Making directories to save results and weights to
    os.makedirs(f'{args.model_dir}/{args.test_no}', exist_ok=True)
    os.makedirs(f'{args.log_dir}/{args.test_no}', exist_ok=True)
    os.makedirs(f'{args.plot_dir}/{args.test_no}', exist_ok=True)

    # Printing out configuration at start of training to make sure all correct
    print(args)
    print('------------------------------------')
    print(f'Model architechture: {args.arch}')
    print(f'Debiasing configuration: {args.debias_config}')
    print(f'Training dataset: {args.dataset}')
    print('------------------------------------')

    os.environ['CUDA_VISIBLE_DEVICES'] = args.CUDA_VISIBLE_DEVICES

    set_seed(seed=args.seed)  # Setting seeds for reproducibility

    main()
