import copy
import random
import torch
from model.pretrain_net import FeatureExtractor, TransformerEncoder, SleepMLP
from utils.config import ModelConfig
import os
from dataloader.data_loader import Builder
from torch.utils.data import DataLoader
import numpy as np
from model.incremental_algorithm import CPC, BufferPseudoLabelFinetune
from utils.util import Evaluator, compute_aaa, compute_forget
import torch.nn.functional as F


def evaluator(model, dl, args):
    if type(model) == tuple:
        model[0].eval()
        model[1].eval()
        model[2].eval()
    else:
        model.eval()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    model[0].to(device)
    model[1].to(device)
    model[2].to(device)

    model_param = ModelConfig(args.dataset)
    y_pred = []
    y_test = []
    predictions = None
    bh = False
    with torch.no_grad():
        for batch_idx, data in enumerate(dl):
            eog, eeg, labels = data[0].to(device), data[1].to(device), data[2].to(device)
            epoch_size = model_param.EpochLength
            eog = eog.view(-1, model_param.EogNum, epoch_size)
            eeg = eeg.view(-1, model_param.EegNum, epoch_size)
            eeg_eog_feature = model[0](eeg, eog)

            eeg_eog_feature = model[1](eeg_eog_feature)

            prediction = model[2](eeg_eog_feature)

            if not bh:
                predictions = prediction
                bh = True
            else:
                predictions = torch.concat((predictions, prediction), dim=0)
            _, predicted = torch.max(prediction.data, dim=1)
            predicted, labels = torch.flatten(predicted), torch.flatten(labels)

            predicted = predicted.tolist()
            y_pred.extend(predicted)
            labels = labels.tolist()
            y_test.extend(labels)
        report = (y_test, y_pred, predictions)
        return report


def get_new_task_loader(args, new_task_idx, is_buffer, shuffle):
    new_task_path = [[], []]
    file_path = args.file_path + f"/{new_task_idx}/data"
    label_path = args.file_path + f"/{new_task_idx}/label"
    num = 0
    while os.path.exists(file_path + f"/{num}.npy"):
        new_task_path[0].append(file_path + f"/{num}.npy")
        new_task_path[1].append(label_path + f"/{num}.npy")
        num += 1
    if is_buffer:
        new_task_builder = Builder(new_task_path, args).BufferDataset
    else:
        new_task_builder = Builder(new_task_path, args).Dataset
    new_task_loader = DataLoader(dataset=new_task_builder, batch_size=args.batch, shuffle=shuffle, num_workers=4)

    return new_task_loader


def trainer(old_task_loader, new_task_idx, args, performance):
    cross_epoch = args.cross_epoch
    num = 1
    print(new_task_idx)
    for new_task_id in new_task_idx:
        print("New Task Id", new_task_id)

        if num >= args.train_num:
            args.alpha = np.power(0.1, np.log10(num/args.train_num)+2)

        new_task_loader = get_new_task_loader(args, new_task_id, False, True)

        feature_extractor = FeatureExtractor(args).float()
        sleep_classifier = SleepMLP(args).float()
        feature_encoder = TransformerEncoder(args).float()

        if not os.path.exists(f"model_parameter/{args.dataset}/"
                              f"EpochNum_{cross_epoch}_{args.algorithm}/individual_{num}"):
            os.makedirs(f"model_parameter/{args.dataset}/"
                        f"EpochNum_{cross_epoch}_{args.algorithm}/individual_{num}")

        if num == 1:
            """
            First Individual for Incremental Learning 
            """
            feature_extractor.load_state_dict(
                torch.load(
                    f"model_parameter/{args.dataset}/Pretrain/feature_extractor_parameter_{args.seed}.pkl"))
            feature_encoder.load_state_dict(
                torch.load(
                    f"model_parameter/{args.dataset}/Pretrain/feature_encoder_parameter_{args.seed}.pkl"))
            sleep_classifier.load_state_dict(
                torch.load(
                    f"model_parameter/{args.dataset}/Pretrain/sleep_classifier_parameter_{args.seed}.pkl"))
            old_task_ans = evaluator((feature_extractor, feature_encoder, sleep_classifier), old_task_loader, args)

            old_task_evaluator = Evaluator(old_task_ans[0], old_task_ans[1])
            old_task_acc, old_task_mf1 = old_task_evaluator.metric_acc(), old_task_evaluator.metric_mf1()
            performance['stability']['ACC'].append(old_task_acc)
            performance['stability']['MF1'].append(old_task_mf1)

            old_task_aaa = compute_aaa(performance['stability']['ACC'])
            old_task_forget = compute_forget(performance['stability']['ACC'])
            performance['stability']['AAA'].append(old_task_aaa)
            performance['stability']['FR'].append(old_task_forget)

        else:
            """
            Load last model
            """
            feature_extractor.load_state_dict(torch.load(
                    f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                    f"/individual_{num-1}/feature_extractor_parameter_{args.seed}.pkl"))
            feature_encoder.load_state_dict(torch.load(
                    f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                    f"/individual_{num-1}/feature_encoder_parameter_{args.seed}.pkl"))
            sleep_classifier.load_state_dict(torch.load(
                    f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                    f"/individual_{num-1}/sleep_classifier_parameter_{args.seed}.pkl"))

        cur_blocks = (feature_extractor, feature_encoder, sleep_classifier)
        teacher_blocks = copy.deepcopy(cur_blocks)
        last_blocks = copy.deepcopy(cur_blocks)
        tmp_blocks, tmp_blocks_teacher = incremental_trainer(cur_blocks, teacher_blocks,
                                                             args, new_task_loader, new_task_id, num)
        """Store Newest Model"""
        state_f = tmp_blocks[0].state_dict()
        for key in state_f.keys():
            state_f[key] = state_f[key].to(torch.device("cpu"))

        state_encoder = tmp_blocks[1].state_dict()
        for key in state_encoder.keys():
            state_encoder[key] = state_encoder[key].to(torch.device("cpu"))

        state_sleep = tmp_blocks[2].state_dict()
        for key in state_sleep.keys():
            state_sleep[key] = state_sleep[key].to(torch.device("cpu"))
        torch.save(state_f,
                   f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                   f"/individual_{num}/feature_extractor_parameter_{args.seed}.pkl")
        torch.save(state_encoder,
                   f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                   f"/individual_{num}/feature_encoder_parameter_{args.seed}.pkl")
        torch.save(state_sleep,
                   f"model_parameter/{args.dataset}/EpochNum_{cross_epoch}_{args.algorithm}"
                   f"/individual_{num}/sleep_classifier_parameter_{args.seed}.pkl")

        """Initial Model"""
        feature_extractor_initial = FeatureExtractor(args).float()
        sleep_classifier_initial = SleepMLP(args).float()
        feature_encoder_initial = TransformerEncoder(args).float()

        feature_extractor_initial.load_state_dict(
            torch.load(
                f"model_parameter/{args.dataset}/Pretrain/feature_extractor_parameter_{args.seed}.pkl"))
        feature_encoder_initial.load_state_dict(
            torch.load(
                f"model_parameter/{args.dataset}/Pretrain/feature_encoder_parameter_{args.seed}.pkl"))
        sleep_classifier_initial.load_state_dict(
            torch.load(
                f"model_parameter/{args.dataset}/Pretrain/sleep_classifier_parameter_{args.seed}.pkl"))

        """Metric"""
        new_task_initial_ans = evaluator((feature_extractor_initial,
                                          feature_encoder_initial,
                                          sleep_classifier_initial), new_task_loader, args)
        new_task_before_ans = evaluator(last_blocks, new_task_loader, args)
        new_task_after_teacher_ans = evaluator(tmp_blocks_teacher, new_task_loader, args)
        new_task_after_ans = evaluator(tmp_blocks, new_task_loader, args)

        new_initial_evaluator = Evaluator(new_task_initial_ans[0], new_task_initial_ans[1])
        new_before_evaluator = Evaluator(new_task_before_ans[0], new_task_before_ans[1])
        new_after_teacher_evaluator = Evaluator(new_task_after_teacher_ans[0], new_task_after_teacher_ans[1])
        new_after_evaluator = Evaluator(new_task_after_ans[0], new_task_after_ans[1])

        new_task_initial_acc, new_task_initial_mf1 = new_initial_evaluator.metric_acc(), new_initial_evaluator.metric_mf1()
        new_task_before_acc, new_task_before_mf1 = new_before_evaluator.metric_acc(), new_before_evaluator.metric_mf1()
        new_task_after_acc, new_task_after_mf1 = new_after_evaluator.metric_acc(), new_after_evaluator.metric_mf1()

        performance['plasticity'][new_task_id]['ACC'] = [new_task_initial_acc, new_task_before_acc, new_task_after_acc]
        performance['plasticity'][new_task_id]['MF1'] = [new_task_initial_mf1, new_task_before_mf1, new_task_after_mf1]

        print(f"=========Incremental Individual {new_task_id}=========")
        print(" ACC Initial                    ", performance['plasticity'][new_task_id]['ACC'][0], "\n",
              "ACC Before                    ", performance['plasticity'][new_task_id]['ACC'][1], "\n",
              "ACC After Contrastive Learning ", new_after_teacher_evaluator.metric_acc(), "\n",
              "ACC After Joint Training       ", performance['plasticity'][new_task_id]['ACC'][2], "\n")

        old_task_ans = evaluator(tmp_blocks, old_task_loader, args)
        old_task_teacher_ans = evaluator(tmp_blocks_teacher, old_task_loader, args)

        old_task_evaluator = Evaluator(old_task_ans[0], old_task_ans[1])
        old_task_evaluator_teacher = Evaluator(old_task_teacher_ans[0], old_task_teacher_ans[1])

        old_task_acc = old_task_evaluator.metric_acc()
        old_task_mf1 = old_task_evaluator.metric_mf1()
        performance['stability']['ACC'].append(old_task_acc)
        performance['stability']['MF1'].append(old_task_mf1)

        old_task_aaa = compute_aaa(performance['stability']['ACC'])
        old_task_forget = compute_forget(performance['stability']['ACC'])
        performance['stability']['AAA'].append(old_task_aaa)
        performance['stability']['FR'].append(old_task_forget)

        print(performance["stability"])
        print("=========Model Generalization Ability=========")
        if num == 1:
            print(" ACC Before                    ", performance['stability']['ACC'][0], "\n",
                  "ACC After Contrastive Learning ", old_task_evaluator_teacher.metric_acc(), "\n",
                  "ACC After Joint Training       ", performance['stability']['ACC'][-1], "\n",
                  "MF1 Before                     ", performance['stability']['MF1'][0], "\n",
                  "MF1 After Contrastive Learning ", old_task_evaluator_teacher.metric_mf1(), "\n",
                  "MF1 After Joint Training       ", performance['stability']['MF1'][-1], "\n")
        else:
            print(" ACC Before                    ", performance['stability']['ACC'][-2], "\n",
                  "ACC After Contrastive Learning ", old_task_evaluator_teacher.metric_acc(), "\n",
                  "ACC After Joint Training       ", performance['stability']['ACC'][-1], "\n",
                  "MF1 Before                     ", performance['stability']['MF1'][-2], "\n",
                  "MF1 After Contrastive Learning ", old_task_evaluator_teacher.metric_mf1(), "\n",
                  "MF1 After Joint Training       ", performance['stability']['MF1'][-1], "\n")

        buffer_single_merge(args, new_task_id, num, tmp_blocks)
        print("Buffer_Length", len(args.train_path[0]), len(args.train_path[1]))
        num += 1


def buffer_single_merge(args, new_task_id, num, tmp_blocks):
    cross_epoch = args.cross_epoch
    new_task_path = [[], []]
    file_path = args.file_path + f"/{new_task_id}/data"
    label_path = args.file_path + f"/{new_task_id}/label"
    idx = 0
    while os.path.exists(file_path + f"/{idx}.npy"):
        new_task_path[0].append(file_path + f"/{idx}.npy")
        new_task_path[1].append(label_path + f"/{idx}.npy")
        idx += 1
    new_task_builder = Builder(new_task_path, args).Dataset
    new_task_loader = DataLoader(dataset=new_task_builder, batch_size=args.batch, shuffle=False, num_workers=4)
    new_task_after_ans = evaluator(tmp_blocks, new_task_loader, args)

    save_path = f"model_parameter/{args.dataset}/" \
                f"EpochNum_{cross_epoch}_{args.algorithm}/individual_{num}/label"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    confident_epoch_n = 15
    confidence_level = 0.9
    new_task_out = new_task_after_ans[2]
    mean_t_pred = torch.softmax(new_task_out, dim=1)
    pred_prob = mean_t_pred.max(1, keepdim=True)[0].squeeze()
    pred_label = mean_t_pred.max(1, keepdim=True)[1].squeeze()

    pred_prob = pred_prob.cpu().numpy()
    pred_label = pred_label.cpu().numpy()
    for bh in range(pred_prob.shape[0]):
        confident_epoch_num_per_seq = np.sum(pred_prob[bh, :] >= confidence_level)
        if confident_epoch_num_per_seq >= confident_epoch_n:
            confident_label = pred_label[bh, :].reshape(-1, 1)
            confident_label = np.squeeze(confident_label)
            save_label_path = save_path + f"/{bh}.npy"
            np.save(save_label_path, confident_label)
            args.train_path[1].append(save_label_path)
            args.train_path[0].append(new_task_path[0][bh])


def incremental_trainer(blocks, teacher_blocks, args, new_task_loader, new_task_id, num):
    cross_epoch = args.cross_epoch
    if args.algorithm == 'cpc':
        contrastive_algorithm = CPC(teacher_blocks, args)
    else:
        contrastive_algorithm = None

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    tmp_blocks = None

    blocks[0].to(device)
    blocks[1].to(device)
    blocks[2].to(device)

    teacher_blocks[0].to(device)
    teacher_blocks[1].to(device)
    teacher_blocks[2].to(device)

    """
    Firstly, train teacher model using contrastive method to obtain confident pseudo-label
    """

    for epoch in range(1, args.ssl_epoch + 1):
        teacher_blocks[0].train()
        teacher_blocks[1].train()
        teacher_blocks[2].train()

        epoch_loss = []

        for batch_idx, data in enumerate(new_task_loader):
            eog, eeg, label = data[0].to(device), data[1].to(device), data[2].to(device)
            loss, tmp_blocks_teacher = contrastive_algorithm.update(eeg, eog, label)
            epoch_loss.append(loss)
        print(f"Incremental Individual ID {int(num)}  Contrastive Epoch {epoch} Loss {np.mean(epoch_loss)}")

    """
    Secondly, using pseudo-label and train loader for joint-training.
    """
    algorithm = BufferPseudoLabelFinetune(blocks, tmp_blocks_teacher, args)

    shuffle = True
    buffer_loader = get_new_task_loader(args, new_task_id, True, shuffle)
    print(f"Incremental Individual ID {int(num)}  Alpha={args.alpha}")
    optimizer = torch.optim.Adam([{"params": list(blocks[0].parameters())},
                                  {"params": list(blocks[1].parameters())}],
                                 lr=args.cl_lr,
                                 betas=(args.beta1, args.beta2),
                                 weight_decay=args.weight_decay)
    model_param = ModelConfig(args.dataset)
    align_feature = []
    for epoch in range(args.incremental_epoch+1):
        blocks[0].train()
        blocks[1].train()
        blocks[2].train()

        tmp_blocks_teacher[0].eval()
        tmp_blocks_teacher[1].eval()
        tmp_blocks_teacher[2].eval()

        epoch_loss = []
        kl_loss = []

        if epoch % cross_epoch == 0:
            align_feature.append([])

        for batch_idx, data in enumerate(buffer_loader):

            eog, eeg, label = data[0].to(device), data[1].to(device), data[2].to(device)
            loss, tmp_blocks, feature_before = algorithm.update(eeg, eog, label)
            if epoch % cross_epoch == 0:
                align_feature[-1].append(feature_before)
            epoch_loss.append(loss)

            if epoch % cross_epoch == 0 and epoch != 0:
                eog_train = eog[:, 20:, :, :].contiguous().view(-1, 2, 3000)
                eeg_train = eeg[:, 20:, :, :].contiguous().view(-1, model_param.EegNum, 3000)
                feature_latter = blocks[0](eeg_train, eog_train)
                feature_latter = blocks[1](feature_latter)
                optimizer.zero_grad()
                z1 = torch.nn.functional.log_softmax(feature_latter, dim=-1)
                z2 = F.softmax(align_feature[-2][batch_idx].detach(), dim=-1)
                loss_clr_new = F.kl_div(z1, z2, reduction='batchmean')
                loss_clr_new.backward()
                optimizer.step()
                kl_loss.append(loss_clr_new.item())

        print(f"Incremental Individual ID {int(num)}  Joint Fine-tuning Epoch {epoch+1} Loss {np.mean(epoch_loss)}")
    return tmp_blocks, tmp_blocks_teacher
