# Time : 2023/11/13 12:52
# Author : 小霸奔
# FileName: pretrain.p
from model.pretrain_net import FeatureExtractor_BCI2000, TransformerEncoder_BCI2000, SleepMLP_BCI2000
import torch.nn as nn
import torch
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from utils.config import ModelConfig
import os


def pretraining_bci2000(train_dl, val_dl, args):
    """
    multi_modality_model  (feature_extractor, att_encoder, sleep_classifier)
    Pretrain
    """
    multi_modality_model = train_block(train_dl, val_dl, args)
    state_f = multi_modality_model[0].state_dict()
    for key in state_f.keys():
        state_f[key] = state_f[key].to(torch.device("cpu"))

    state_encoder = multi_modality_model[1].state_dict()
    for key in state_encoder.keys():
        state_encoder[key] = state_encoder[key].to(torch.device("cpu"))

    state_sleep = multi_modality_model[2].state_dict()
    for key in state_sleep.keys():
        state_sleep[key] = state_sleep[key].to(torch.device("cpu"))

    if not os.path.exists(f"model_parameter/{args['dataset']}/Pretrain"):
        os.makedirs(f"model_parameter/{args['dataset']}/Pretrain")

    torch.save(state_f,
               f"model_parameter/{args.dataset}/Pretrain/feature_extractor_parameter_{args['rand']}.pkl")
    torch.save(state_encoder,
               f"model_parameter/{args.dataset}/Pretrain/feature_encoder_parameter_{args['rand']}.pkl")
    torch.save(state_sleep,
               f"model_parameter/{args.dataset}/Pretrain/sleep_classifier_parameter_{args['rand']}.pkl")


def train_block(train_dl, val_dl, args):
    """
    :param train_dl: train set dataloader
    :param val_dl: val set dataloader
    :param args: train parameters
    :return: best_modality_feature: Net work structure in best epoch
    """
    # Initialize parameter
    device = args.device
    total_acc = []
    total_f1 = []
    best_acc = 0
    best_f1 = 0
    best_epoch = 0
    # Build model network

    feature_extractor = FeatureExtractor_BCI2000(args).float().to(device)
    sleep_classifier = SleepMLP_BCI2000(args).float().to(device)
    feature_encoder = TransformerEncoder_BCI2000(args).float().to(device)

    # loss function
    classifier_criterion = nn.CrossEntropyLoss().to(device)

    # optimizer
    optimizer_encoder = torch.optim.Adam(list(feature_extractor.parameters())
                                         + list(sleep_classifier.parameters())
                                         + list(feature_encoder.parameters()), lr=args["lr"],
                                         betas=(args.beta[0], args.beta[1]),
                                         weight_decay=args.weight_decay)

    model_param = ModelConfig(args.dataset)


    for epoch in range(1, args["pretrain_epoch"] + 1):
        print(f"--------------------Epoch{epoch}---Pretrain-----------------------------")
        feature_extractor.train()
        sleep_classifier.train()
        feature_encoder.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(train_dl):
            x, label = data[0].to(device), data[1].to(device)
            # epoch_size = model_param.EpochLengthFace  # batch, 32, 7500
            ff = feature_extractor(x)
            ff = feature_encoder(ff)
            pred = sleep_classifier(ff)
            # Compute Classification Loss
            # print(pred.shape, label.shape)
            loss_classifier = classifier_criterion(pred, label.long())

            optimizer_encoder.zero_grad()
            loss_classifier.backward()
            torch.nn.utils.clip_grad_norm_(sleep_classifier.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(feature_extractor.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(feature_encoder.parameters(), 1.0)
            optimizer_encoder.step()

            running_loss += loss_classifier.item()

            # if batch_idx % 10 == 9:  # 输出每次的平均loss
            print('\n [%d,  %5d] total_loss: %.3f ' % (epoch, batch_idx + 1, running_loss))
            running_loss = 0.0


        if epoch % 1 == 0:
            print(f" -------------------Epoch{epoch}---Val------------------------------")
            report = dev_block((feature_extractor, feature_encoder, sleep_classifier),
                               val_dl, args, model_param)
            total_acc.append(report[0])
            total_f1.append(report[1])

        if total_acc[-1] > best_acc:
            best_acc = total_acc[-1]
            best_f1 = total_f1[-1]
            best_epoch = epoch
            best_modality_feature = (feature_extractor, feature_encoder, sleep_classifier)
        print("dev_acc:", total_acc)
        print("dev_macro_f1:", total_f1)
    else:
        print(f"Pretraining: Best Epoch:{best_epoch}  Best ACC:{best_acc}  Best F1:{best_f1}")

    return best_modality_feature


def dev_block(model, val_dl, args, model_param):
    """
    :param model: (feature_extractor, att_encoder, sleep_classifier)
    :param val_dl: Val Set Dataloader
    :param args: Val parameters
    :param model_param: Model Parameters
    :return: report: tuple(acc, macro_f1)
    """
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2].eval()
    else:
        model.eval()

    device = args.device
    criterion = nn.CrossEntropyLoss()

    y_pred = []
    y_test = []
    with torch.no_grad():
        correct = 0.0
        total = 0.0
        dev_mean_loss = 0.0
        for batch_idx, data in enumerate(val_dl):
            x, labels = data[0].to(device), data[1].to(device)
            epoch_size = model_param.EpochLengthBCI2000

            x = x.view(-1, model_param.BCICn, epoch_size)

            ff = model[0](x)

            ff = model[1](ff)

            pred = model[2](ff)

            dev_loss = criterion(pred, labels.long())
            dev_mean_loss += dev_loss.item()

            _, predicted = torch.max(pred.data, dim=1)
            predicted, labels = torch.flatten(predicted), torch.flatten(labels)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            acc = correct / total
            count = batch_idx
            predicted = predicted.tolist()
            y_pred.extend(predicted)
            labels = labels.tolist()
            y_test.extend(labels)

        macro_f1 = f1_score(y_test, y_pred, average="macro")
        print('dev loss:', dev_mean_loss / count, 'Accuracy on Physionet-MI:', acc, 'F1 score on Physionet-MI:', macro_f1, )
        print(classification_report(y_test, y_pred, target_names=model_param.ClassNamesBCI2000))
        confusion_mtx = confusion_matrix(y_test, y_pred)
        print(confusion_mtx)

        report = (acc, macro_f1)
        return report
