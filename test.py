import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, roc_auc_score, f1_score, precision_score, accuracy_score
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset import *
from models import *
from globalbaz import args, DP, device
from train_epoch_variations import *
import torch
import torchvision.transforms as transforms


# Inverse transform to get normalize image back to original form for visualization
inv_normalize = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    std=[1/0.229, 1/0.224, 1/0.255]
)


def test(index, df, mel_idx, transforms_val):
    # Setting test sets based on arguments
    if args.tune:
        test_names = ['val_data']
    elif args.heid_test_marked:
        test_names = ['blank', 'marked']
    elif args.heid_test_rulers:
        test_names = ['blank', 'rulers']
    else:
        test_names = ['AtlasDerm', 'AtlasClinic', 'ASAN', 'MClassD', 'MClassC']

    n_test = 8  # Number of tests for test-time augmentation
    dataset_test = SIIMISICDataset(df, 'test', 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers)
    # Setting different number of units for fully connected layer based on feature extractor output
    in_ch = 2048
    if args.META:
        n_meta_features = 583
        in_ch = in_ch + 128 # Output size of MetaNN is 128 and output size of Resnext is 2048

    # Defining feature extractor based on args

    model_encoder = ResNext101()
    
    model_classifier = ClassificationHead(out_dim=args.out_dim, in_ch=in_ch)  # Defining classification head
    # if DP:  # Parallelising if number of GPUs allows
    #     model_encoder = nn.DataParallel(model_encoder)
    #     model_classifier = nn.DataParallel(model_classifier)
    model_encoder = model_encoder.to(device)  # Sending to GPU
    
    model_classifier = model_classifier.to(device)  # Sending to GPU
    # Loading weights (getting rid of module prefix from dataparallel if present)
    checkpoint_encoder_dp = torch.load(f'{args.model_dir}/{args.test_no}/encoder_all_data_Test{args.test_no}.pth')
    encoder_state_dict_dp = checkpoint_encoder_dp['model_state_dict']
    encoder_state_dict = {key.replace("module.", ""): value for key, value in encoder_state_dict_dp.items()}
    model_encoder.load_state_dict(encoder_state_dict)


    classifier_state_dict_dp = torch.load(f'{args.model_dir}/{args.test_no}/classifier_all_data_Test{args.test_no}.pth')
    classifier_state_dict = {key.replace("module.", ""): value for key, value in classifier_state_dict_dp.items()}
    model_classifier.load_state_dict(classifier_state_dict, strict=True)
    
    model_encoder.eval()
    
    model_classifier.eval()
    
    
    if args.META:
        meta_encoder = MetaNN(n_meta_features)
        meta_encoder = meta_encoder.to(device)
        
        meta_state_dict_dp = torch.load(f'{args.model_dir}/{args.test_no}/meta_all_data_Test{args.test_no}.pth')
        meta_state_dict = {key.replace("module.", ""): value for key, value in meta_state_dict_dp.items()}
        meta_encoder.load_state_dict(meta_state_dict) 
        
        meta_encoder.eval()

    LOGITS = []
    PROBS = []
    TARGETS = []

    with torch.no_grad():
        if not args.META:    
            for (data, target) in tqdm(test_loader):  # Getting test data and labels
                data, target = data.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                # Testing 8 times (test-time augmentation)
                for I in range(n_test):
                    feat_out = model_encoder(get_trans(data, I))
                    l = model_classifier(feat_out)
                    logits += l
                    probs += torch.sigmoid(l)
                logits /= n_test
                probs /= n_test
    
                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())
        else:
            for (data, meta,target) in tqdm(test_loader):  # Getting test data and labels
                data, meta, target = data.to(device), meta.to(device), target.to(device)
                logits = torch.zeros((data.shape[0], args.out_dim)).to(device)
                probs = torch.zeros((data.shape[0], args.out_dim)).to(device)
                # Testing 8 times (test-time augmentation)
                for I in range(n_test):
                    feat_out = model_encoder(get_trans(data, I))
                    meta_out = meta_encoder(meta)
                    l = model_classifier(feat_out,meta_out)
                    logits += l
                    probs += torch.sigmoid(l)
                logits /= n_test
                probs /= n_test
    
                LOGITS.append(logits.detach().cpu())
                PROBS.append(probs.detach().cpu())
                TARGETS.append(target.detach().cpu())

    LOGITS = torch.cat(LOGITS).numpy()
    TARGETS = torch.cat(TARGETS).numpy()
    PROBS = torch.cat(PROBS).numpy()

    MALIGNANT_PRED = np.round(PROBS)

    acc = accuracy_score(TARGETS, MALIGNANT_PRED)  # Calculating accuracy, 0.5 threshold
    a_u_c = roc_auc_score(TARGETS, PROBS)  # Calculating AUC

    cm = confusion_matrix(TARGETS, MALIGNANT_PRED)  # Defining confusion matrix
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]  # Getting confusion matrix values
    sensitivity = tp / (tp + fn)  # calculating sensitivity
    specificity = tn / (tn + fp)  # calculating specificity
    # precision = precision_score(TARGETS, MALIGNANT_PRED)  # Calculating precision at 0.5 thresh
    f1 = f1_score(TARGETS, MALIGNANT_PRED)  # Calculating f1 score
    results = (f'Test: {args.test_no}, test data: {test_names[index]}, acc: {acc}, AUC: {a_u_c},'
               f' Sensitivity/Recall (0.5 thresh): {sensitivity}, Specificity (0.5 thresh): {specificity},'
               f'f1 score: {f1}')

    # writing metrics to text file
    with open(os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'), 'a') as appender:
        appender.write(results + '\n')

    # printing results to view during testing
    print('---------------------------------------------------------------------------------')
    print(f'Test: {args.test_no}, test data: {test_names[index]}, acc: {acc}, AUC: {a_u_c}')
    print('---------------------------------------------------------------------------------')
    print(f"More complete results saved to: " + os.path.join(args.log_dir, f'{args.test_no}/log_Test{args.test_no}.txt'))
    print('---------------------------------------------------------------------------------')

    # # Formatting confusion matrix into heatmap.
    # # Labels manually specified due to formatting issues
    # plt.figure()
    # labels = cm
    # sns.heatmap(cm, annot=labels, fmt='')
    # plt.xlabel('Predicted')
    # plt.ylabel('Truth')
    # plt.title(f'Confusion Matrix Test{args.test_no} ({test_names[index]})');
    # plt.savefig(f'./{args.plot_dir}/{args.test_no}/{test_names[index]}_CM.png', dpi=300)
    # plt.clf()

    fpr, tpr, thresholds = roc_curve(TARGETS == mel_idx, PROBS)  # We can use these to plot custom ROC curves

    return fpr, tpr, a_u_c, sensitivity, specificity


# Testing for cross validation
def cv_scores(df_train, mel_idx, transforms_val, transforms_marked):
    PROBS = []
    dfs = []
    acc_cv = []

    for fold in range(5):  # Looping through folds
        i_fold = fold

        # Loading validation data based on which fold is left out
        df_valid = df_train[df_train['fold'] == i_fold]
        dataset_valid = SIIMISICDataset(df_valid, 'train', 'val', transform=transforms_val, transform2=transforms_marked)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=args.batch_size,
                                                   num_workers=args.num_workers, drop_last=False)
        # Setting different number of units for fully connected layer based on feature extractor output
        if args.arch == 'resnet101' or args.arch == 'resnext101' or args.arch == 'inception':
            in_ch = 2048
        elif args.arch == 'densenet':
            in_ch = 2208
        else:
            in_ch = 1536

        # Defining feature extractor model
        if args.arch == 'enet':
            model_encoder = enetv2(args.enet_type)
        if args.arch == 'resnet101':
            model_encoder = ResNet101()
        if args.arch == 'resnext101':
            model_encoder = ResNext101()
        if args.arch == 'densenet':
            model_encoder = DenseNet()
        if args.arch == 'inception':
            model_encoder = Inception()

        model_classifier = ClassificationHead(out_dim=args.out_dim, in_ch=in_ch)  # Defining main classifier head
        # if DP:  # Parallelising if number of GPUs allows
        #     model_encoder = nn.DataParallel(model_encoder)
        #     model_classifier = nn.DataParallel(model_classifier)
        model_encoder = model_encoder.to(device)  # Sending to GPU
        model_classifier = model_classifier.to(device)  # Sending to GPU
        # Loading weights (getting rid of module prefix from dataparallel if present)
        checkpoint_encoder_dp = torch.load(f'{args.model_dir}/{args.test_no}/encoder_all_data_Test{args.test_no}.pth')
        encoder_state_dict_dp = checkpoint_encoder_dp['model_state_dict']
        encoder_state_dict = {key.replace("module.", ""): value for key, value in encoder_state_dict_dp.items()}
        model_encoder.load_state_dict(encoder_state_dict)

        classifier_state_dict_dp = torch.load(
            f'{args.model_dir}/{args.test_no}/classifier_all_data_Test{args.test_no}.pth')
        classifier_state_dict = {key.replace("module.", ""): value for key, value in classifier_state_dict_dp.items()}
        model_classifier.load_state_dict(classifier_state_dict, strict=True)
        model_encoder.eval()
        model_classifier.eval()

        criterion, _, _ = criterion_func(df_train)  # Getting loss function

        this_PROBS, TARGETS = val_epoch(model_encoder, model_classifier, valid_loader, criterion, n_test=8, get_output=True)
        PROBS.append(this_PROBS)
        dfs.append(df_valid)
        acc_cv.append((this_PROBS.argmax(1) == TARGETS).mean() * 100.)

    dfs = pd.concat(dfs).reset_index(drop=True)
    dfs['pred'] = np.concatenate(PROBS).squeeze()

    # Cross val accuracy
    cv_acc = np.mean(acc_cv)

    # Raw auc_all
    cv_auc = roc_auc_score(dfs['target'] == mel_idx, dfs['pred'])

    # Rank per fold auc_all
    dfs2 = dfs.copy()
    for i in range(5):
        dfs2.loc[dfs2['fold'] == i, 'pred'] = dfs2.loc[dfs2['fold'] == i, 'pred'].rank(pct=True)
    cv_auc_rpf = roc_auc_score(dfs2['target'] == mel_idx, dfs2['pred'])

    return cv_acc, cv_auc, cv_auc_rpf

# Plotting ROC curves of results in style of ggplot
def ROC_curve(roc_plt_lst):
    # Setting test data based on args passed
    if args.tune:
        test_names = ['val_data']
    elif args.heid_test_marked:
        test_names = ['plain', 'marked']
    elif args.heid_test_rulers:
        test_names = ['plain', 'rulers']
    else:
        test_names = ['AtlasDerm', 'AtlasClinic', 'ASAN', 'MClassD', 'MClassC']
    # Plot ROC curve
    fig = plt.figure()
    plt.style.use('ggplot')
    plt.use_sticky_edges = False
    plt.margins(0.005)
    for index, thruple in enumerate(roc_plt_lst):
        fpr, tpr, a_u_c, _, _ = thruple
        plt.plot(fpr, tpr, label=f'{test_names[index]} (area = {round(a_u_c, 3)})')
        plt.plot([0, 1], [0, 1], 'k--')  # random predictions line
        plt.xlabel('False Positive Rate (1-specificity)', fontsize=14)
        plt.ylabel('True Positive Rate (sensitivity)', fontsize=14)
        plt.title(f'Receiver Operating Characteristic Test{args.test_no}', fontsize=14)
        plt.legend(loc="lower right")
        plt.rc('legend', fontsize=10)  # legend fontsize
    plt.show()
    fig.savefig(f'./{args.plot_dir}/{args.test_no}/ROC_curve.pdf')
