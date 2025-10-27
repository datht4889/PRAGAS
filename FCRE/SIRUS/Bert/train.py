import argparse
import torch
import random
import sys
import copy
import numpy as np
import torch.nn as nn
import torch.optim as optim
from sklearn.cluster import KMeans
from config import Config
import torch.nn.functional as F
from sklearn.cluster import AgglomerativeClustering
from sam import *

import warnings
warnings.filterwarnings("ignore")


from sampler import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from mixup import mixup_data_augmentation
from encoder import EncodingModel
# import wandb
import copy
from transformers import BertTokenizer
from losses import TripletLoss, RKD, KLDivAndAngleLoss, MultipleNegativesRankingLoss, NegativeCosSimLoss

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        print("SIRIUS BERT")
        print(f'SAM: {config.sam}')
        print(f'SAM_type: {config.sam_type}')
        print(f'SAM Optimizer: {config.sam_optimizer}')
        print(f'decay: {config.decay}')

        
    def _edist(self, x1, x2):
        '''
        input: x1 (B, H), x2 (N, H) ; N is the number of relations
        return: (B, N)
        '''
        b = x1.size()[0]
        L2dist = nn.PairwiseDistance(p=2)
        dist = [] # B
        for i in range(b):
            dist_i = L2dist(x2, x1[i])
            dist.append(torch.unsqueeze(dist_i, 0)) # (N) --> (1,N)
        dist = torch.cat(dist, 0) # (B, N)
        return dist
    # def _cosine_similarity(self, x1, x2):
    #     '''
    #     input: x1 (B, H), x2 (N, H) ; N is the number of relations
    #     return: (B, N)
    #     '''
    #     b = x1.size()[0]
    #     cos = nn.CosineSimilarity(dim=1)
    #     sim = []
    #     for i in range(b):
    #         sim_i = cos(x2, x1[i])
    #         sim.append(torch.unsqueeze(sim_i, 0))
    #     sim = torch.cat(sim, 0)
    #     return sim
    
    def _cosine_similarity(self, x1, x2):

        x1_norm = F.normalize(x1, p=2, dim=1)  # (B, H)
        x2_norm = F.normalize(x2, p=2, dim=1)  # (N, H)

        sim = torch.matmul(x1_norm, x2_norm.T)  # (B, N)

        return sim


    def get_memory_proto(self, encoder, dataset):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset):
        '''
        only for one relation data
        '''
        N, M = len(dataset), self.config.memory_size
        data_loader = get_data_loader_BERT(self.config, dataset, shuffle=False, \
            drop_last= False, batch_size=1) # batch_size must = 1
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)

        features = np.concatenate(features) # tensor-->numpy array; (N, H)
        
        if N <= M: 
            return copy.deepcopy(dataset), torch.from_numpy(features)

        num_clusters = M # memory_size < len(dataset)
        distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features) # (N, M)

        mem_set = []
        mem_feas = []
        for k in range(num_clusters):
            sel_index = np.argmin(distances[:, k])
            sample = dataset[sel_index]
            mem_set.append(sample)
            mem_feas.append(features[sel_index])

        mem_feas = np.stack(mem_feas, axis=0) # (M, H)
        mem_feas = torch.from_numpy(mem_feas)
        # proto = memory mean
        # rel_proto = mem_feas.mean(0)
        # proto = all mean
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas
        # return mem_set, features, rel_proto
        
    
    def get_cluster_and_centroids(self, embeddings):

        clustering_model = AgglomerativeClustering(n_clusters=None,metric="cosine",linkage="average", distance_threshold=self.config.distance_threshold).fit(embeddings)
        clusters = clustering_model.fit_predict(embeddings)
        centroids = {}
        for cluster_id in np.unique(clusters):
            if cluster_id not in centroids:
                cluster_embeddings = embeddings[clusters == cluster_id]
                centroid = torch.mean(cluster_embeddings, dim=0)
                centroids[cluster_id] = centroid

        return clusters, centroids

    def train_model(self, encoder, old_encoder, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)

        optimizer = optim.AdamW(params=encoder.parameters(), lr=self.config.lr, weight_decay=self.config.decay)
        if self.config.SAM:
            base_optimizer = optim.AdamW
            if self.config.sam_optimizer=='SAM':
                optimizer = SAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='ASAM':
                optimizer = ASAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='ESAM':
                optimizer = ESAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='GCSAM':
                optimizer = GCSAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='FriendlySAM':
                optimizer = FriendlySAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='LookbehindASAM':
                optimizer = LookbehindASAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='LookbehindSAM':
                optimizer = LookbehindSAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
        encoder.train()
        epoch = self.config.epoch_mem if is_memory else self.config.epoch

        triplet = TripletLoss()
        optimizer.zero_grad()

        relation_2_cluster = {}
        rep_seen_des = []
        relationid2_clustercentroids = {}

        if old_encoder is not None and self.config.distill and self.config.distill_type != 'none':
            self.distill_loss_list = []
            if self.config.distill_type == 'RKD':
                distill_loss_fn = RKD(device=self.config.device)
            elif self.config.distill_type == 'KLDivAndAngleLoss':
                distill_loss_fn = KLDivAndAngleLoss(device=self.config.device)

        for i in range(epoch):         
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                for k in instance.keys():
                    instance[k] = instance[k].to(self.config.device)

                batch_instance = {'ids': [], 'mask': []} 

                batch_instance['ids'] = torch.tensor([seen_des[self.id2rel[label.item()]]['ids'] for label in labels]).to(self.config.device)
                batch_instance['mask'] = torch.tensor([seen_des[self.id2rel[label.item()]]['mask'] for label in labels]).to(self.config.device)

                
                hidden, outputs_words, topk_hidden_indices = encoder(instance, is_distill=True, top_k=self.config.distill_top_k)
                rep_des = encoder(batch_instance, is_des = True) # b, dim

                with torch.no_grad():
                    rep_seen_des = []
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                            'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                        }
                        hidden_des = encoder(sample, is_des=True)
                        hidden_des = hidden_des.detach().cpu().data
                        rep_seen_des.append(hidden_des)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1

                # print(clusters)

                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                relation_2_cluster = {}

                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=self.config.temperature,relation_2_cluster=relation_2_cluster)  # Recompute loss2

                    
                cluster_centroids = []

                for label in labels:
                    cluster_centroids.append(relationid2_clustercentroids[label.item()])

                cluster_centroids  = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
                
                nearest_cluster_centroids = []
                for hid in hidden:
                    cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids, dim=1)

                    try:
                        top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)

                        if len(top2_indices) > 1:
                            top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                        else:
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    except RuntimeError as e:
                        print(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)

                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)

                if flag == 0:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)

                    loss = self.config.lambda_1*loss1 + self.config.lambda_2*loss2 + self.config.lambda_3*loss3

                else:
                    loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss = self.config.lambda_1*loss1 + self.config.lambda_2*loss2  

                if old_encoder is not None and self.config.distill and self.config.distill_type != 'none':
                    old_hidden, old_outputs_words, old_topk_hidden_indices = old_encoder(instance, is_distill=True, top_k=self.config.distill_top_k)
                    old_topk_hidden = torch.gather(old_outputs_words, dim=1, index=old_topk_hidden_indices)  # (B, k, H)
                    topk_hidden = torch.gather(outputs_words, dim=1, index=topk_hidden_indices)  # (B, k, H)
                    if self.config.distill_type in ['RKD', 'KLDivAndAngleLoss']:
                        distill_loss = distill_loss_fn(topk_hidden, old_topk_hidden)
                        # loss = loss + distill_loss * self.config.distill_loss_weight
                        self.distill_loss_list.append(distill_loss.item())
                    else:
                        raise NotImplementedError("Distill Loss {} not implemented".format(self.config.distill_type))

                if not self.config.SAM:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    if self.distill_loss_list != [] and self.config.dynamic_rho:
                        mean_distill_loss = sum(self.distill_loss_list)/len(self.distill_loss_list)
                        distillation_rho = max(0.03, min(mean_distill_loss * self.config.rho_weight, 0.07))
                        # print("Setting rho to: ", distillation_rho)
                        # print("Setting rho to: ", distillation_rho)
                        optimizer.first_step(zero_grad=True, rho=distillation_rho)
                    else:
                        # print("Using config rho: ", self.config.rho)
                        optimizer.first_step(zero_grad=True, rho=self.config.rho)

                    hidden= encoder(instance) # b, dim
                    rep_des = encoder(batch_instance, is_des = True) # b, dim

                    with torch.no_grad():
                        rep_seen_des = []
                        for i2 in range(len(list_seen_des)):
                            sample = {
                                'ids' : torch.tensor([list_seen_des[i2]['ids']]).to(self.config.device),
                                'mask' : torch.tensor([list_seen_des[i2]['mask']]).to(self.config.device)
                            }
                            hidden_des = encoder(sample, is_des=True)
                            hidden_des = hidden_des.detach().cpu().data
                            rep_seen_des.append(hidden_des)
                        rep_seen_des = torch.cat(rep_seen_des, dim=0)
                        clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)
                    flag = 0
                    if len(clusters) == max(clusters) + 1:
                        flag = 1

                    # print(clusters)

                    relationid2_clustercentroids = {}
                    for index, rel in enumerate(seen_relations):
                        relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                    relation_2_cluster = {}

                    for i1 in range(len(seen_relations)):
                        relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                    loss2 = self.moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=self.config.temperature,relation_2_cluster=relation_2_cluster)  # Recompute loss2

                        
                    cluster_centroids = []

                    for label in labels:
                        cluster_centroids.append(relationid2_clustercentroids[label.item()])

                    cluster_centroids  = torch.stack(cluster_centroids, dim = 0).to(self.config.device)
                    
                    nearest_cluster_centroids = []
                    for hid in hidden:
                        cos_similarities = torch.nn.functional.cosine_similarity(hid.unsqueeze(0), cluster_centroids, dim=1)

                        try:
                            top2_similarities, top2_indices = torch.topk(cos_similarities, k=2, dim=0)

                            if len(top2_indices) > 1:
                                top2_centroids = relationid2_clustercentroids[labels[top2_indices[1].item()].item()]
                            else:
                                top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                        except RuntimeError as e:
                            print(f"RuntimeError in top-k selection: {e}")
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                        nearest_cluster_centroids.append(top2_centroids)

                    nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)

                    if flag == 0:
                        loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                        loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)

                        loss = self.config.lambda_1*loss1 + self.config.lambda_2*loss2 + self.config.lambda_3*loss3

                    else:
                        loss1 = self.moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                        loss = self.config.lambda_1*loss1 + self.config.lambda_2*loss2

                    if old_encoder is not None and self.config.distill and self.config.distill_type != 'none':
                        old_hidden, old_outputs_words, old_topk_hidden_indices = old_encoder(instance, is_distill=True, top_k=self.config.distill_top_k)
                        old_topk_hidden = torch.gather(old_outputs_words, dim=1, index=old_topk_hidden_indices)  # (B, k, H)
                        topk_hidden = torch.gather(outputs_words, dim=1, index=topk_hidden_indices)  # (B, k, H)
                        if self.config.distill_type in ['RKD', 'KLDivAndAngleLoss']:
                            distill_loss = distill_loss_fn(topk_hidden, old_topk_hidden)
                            # loss = loss + distill_loss * self.config.distill_loss_weight
                        else:
                            raise NotImplementedError("Distill Loss {} not implemented".format(self.config.distill_type))
                        
                    loss.backward()
                    optimizer.second_step(zero_grad=True)

                # update moment
                if is_memory:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=True)
                    # self.moment.update_allmem(encoder)
                else:
                    self.moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=False)

                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        print('')             

    def train_model_mixup(self, encoder, training_data):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        
        optimizer = optim.AdamW(params=encoder.parameters(), lr=self.config.lr)
        if self.config.SAM:
            base_optimizer = optim.AdamW
            if self.config.sam_optimizer=='SAM':
                optimizer = SAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='ASAM':
                optimizer = ASAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='ESAM':
                optimizer = ESAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='GCSAM':
                optimizer = GCSAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='FriendlySAM':
                optimizer = FriendlySAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='LookbehindASAM':
                optimizer = LookbehindASAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
            elif self.config.sam_optimizer=='LookbehindSAM':
                optimizer = LookbehindSAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, lr=self.config.lr, weight_decay=self.config.decay, betas=(0.9, 0.999))
        encoder.train()
        epoch = 2
        
        loss_retrieval = MultipleNegativesRankingLoss()
        neg_cos_sim_loss = NegativeCosSimLoss()
        
        
        for i in range(epoch):
            for batch_num, (instance, labels, ind) in enumerate(data_loader):
                label_first = [temp[0] for temp in labels]
                label_second = [temp[1] for temp in labels]
                mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance)
                n = len(label_first)
                m = len(label_second)
                new_matrix_labels = np.zeros((n, m), dtype=float)

                # Fill the matrix according to the label comparison
                for i1 in range(n):
                    for j in range(m):
                        if label_first[i1] == label_second[j]:
                            new_matrix_labels[i1][j] = 1.0

                new_matrix_labels_tensor = torch.tensor(new_matrix_labels).to(config.device)
                loss1 = neg_cos_sim_loss(mask_hidden_1, mask_hidden_2)
                
                mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                
                matrix_labels_tensor_mean_12 = np.zeros((mask_hidden_mean_12.shape[0], mask_hidden_mean_12.shape[0]), dtype=float)
                for i1 in range(mask_hidden_mean_12.shape[0]):
                        for j1 in range(mask_hidden_mean_12.shape[0]):
                            if i1 != j1:
                                if label_first[i1] in [label_first[j1], label_second[j1]] and label_second[i1] in [label_first[j1], label_second[j1]]:
                                    matrix_labels_tensor_mean_12[i1][j1] = 1.0
                matrix_labels_tensor_mean_12 = torch.tensor(matrix_labels_tensor_mean_12).to(config.device)
                
                loss2 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                
                
                merged_hidden = torch.cat((mask_hidden_1, mask_hidden_2), dim=0)
                merged_labels = torch.cat((torch.tensor(label_first), torch.tensor(label_second)), dim=0)
                
    
                # if merged_hidden.shape[1] != 4096: # hard code :) for llm2vec
                #     print('something wrong')
                #     logger.info('something wrong')
                #     continue
                loss = self.moment.contrastive_loss(merged_hidden, merged_labels, is_memory = True)
                sum_loss = 0.0
                if not torch.isnan(loss1).any():
                    sum_loss += self.config.mixup_loss_1*loss1
                if not torch.isnan(loss2).any():
                    sum_loss += self.config.mixup_loss_2*loss2
                if not torch.isnan(loss).any():
                    sum_loss += 0.5*loss
                
                if not self.config.SAM:
                    optimizer.zero_grad()
                    sum_loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True)
                    if self.config.use_llm:
                        mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance['input'])
                    else:
                        mask_hidden_1, mask_hidden_2 = encoder.forward_mixup(instance)
                    loss1 = neg_cos_sim_loss(mask_hidden_1, mask_hidden_2)
                    mask_hidden_mean_12 = (mask_hidden_1 + mask_hidden_2) / 2
                    loss2 = loss_retrieval(mask_hidden_mean_12, mask_hidden_mean_12, matrix_labels_tensor_mean_12)
                    
                    
                    merged_hidden = torch.cat((mask_hidden_1, mask_hidden_2), dim=0)
                    merged_labels = torch.cat((torch.tensor(label_first), torch.tensor(label_second)), dim=0)
                    
        
                    # if merged_hidden.shape[1] != 4096: # hard code :)
                    #     print('something wrong')
                    #     logger.info("something wrong")
                    #     continue
                    loss = self.moment.contrastive_loss(merged_hidden, merged_labels, is_memory = True)
                    sum_loss = 0.0
                    if not torch.isnan(loss1).any():
                        sum_loss += self.config.mixup_loss_1*loss1
                    if not torch.isnan(loss2).any():
                        sum_loss += self.config.mixup_loss_2*loss2
                    if not torch.isnan(loss).any():
                        sum_loss += 0.5*loss
                    sum_loss.backward()
                    optimizer.second_step(zero_grad=True)
                    
                        
                self.moment.update(ind, mask_hidden_1.detach().cpu().data, is_memory=True)
                sys.stdout.write('MixupTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, sum_loss.item()) + '\r')
                # logger.info('MixupTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, sum_loss.item()))
                sys.stdout.flush() 
        print('')   

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            hidden = encoder(instance)
            fea = hidden.cpu().data # place in cpu to eval
            logits = -self._edist(fea, seen_proto) # (B, N) ;N is the number of seen relations

            cur_index = torch.argmax(logits, dim=1) # (B)
            pred =  []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size
            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   '\
                .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.flush()        
        print('')
        return corrects / total
    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
        """
        Args:
            encoder: Encoder
            seen_proto: seen prototypes. NxH tensor
            seen_relid: relation id of protoytpes
            test_data: test data
            rep_des: representation of seen relation description. N x H tensor

        Returns:

        """
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        corrects = 0.0
        corrects1 = 0.0
        corrects2 = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                instance[k] = instance[k].to(self.config.device)
            with torch.no_grad():
                hidden = encoder(instance)
            fea = hidden.cpu().data  # place in cpu to eval
            # logits = -self._edist(fea, seen_proto)  # (B, N) ;N is the number of seen relations
            logits = self._cosine_similarity(fea, seen_proto)  # (B, N)
            logits_des = self._cosine_similarity(fea, rep_des)  # (B, N)

            logits_rrf = logits + logits_des 
           
            cur_index = torch.argmax(logits, dim=1)  # (B)
            pred = []
            for i in range(cur_index.size()[0]):
                pred.append(seen_relid[int(cur_index[i])])
            pred = torch.tensor(pred)

            correct = torch.eq(pred, label).sum().item()
            acc = correct / batch_size
            corrects += correct
            total += batch_size

            # by logits_des
            cur_index1 = torch.argmax(logits_des,dim=1)
            pred1 = []
            for i in range(cur_index1.size()[0]):
                pred1.append(seen_relid[int(cur_index1[i])])
            pred1 = torch.tensor(pred1)
            correct1 = torch.eq(pred1, label).sum().item()
            acc1 = correct1/ batch_size
            corrects1 += correct1

            # by rrf
            cur_index2 = torch.argmax(logits_rrf,dim=1)
            pred2 = []
            for i in range(cur_index2.size()[0]):
                pred2.append(seen_relid[int(cur_index2[i])])
            pred2 = torch.tensor(pred2)
            correct2 = torch.eq(pred2, label).sum().item()
            acc2 = correct2/ batch_size
            corrects2 += correct2

            

            sys.stdout.write('[EVAL] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc, 100 * (corrects / total)) + '\r')
            sys.stdout.write('[EVAL DES] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc1, 100 * (corrects1 / total)) + '\r')
            sys.stdout.write('[EVAL RRF] batch: {0:4} | acc: {1:3.2f}%,  total acc: {2:3.2f}%   ' \
                             .format(batch_num, 100 * acc2, 100 * (corrects2 / total)) + '\r')
            sys.stdout.flush()
        print('')
        return corrects / total, corrects1 / total, corrects2 / total

    def _get_sample_text(self, data_path, index):
        sample = {}
        with open(data_path, 'r') as f:
            for i, line in enumerate(f):
                if i == index:
                    items = line.strip().split('\t')
                    sample['relation'] = self.id2rel[int(items[0])-1]
                    sample['tokens'] = items[2]
                    sample['h'] = items[3]
                    sample['t'] = items[5]
        return sample

    def _read_description(self, r_path):
        rset = {}
        with open(r_path, 'r') as f:
            for line in f:
                items = line.strip().split('\t')
                rset[items[1]] = items[2]
        return rset


    def train(self):
        # sampler 
        sampler = data_sampler_CFRL(config=self.config, seed=self.config.seed)
        self.config.vocab_size = sampler.config.vocab_size

        print('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        # encoder
        encoder = EncodingModel(self.config)
        old_encoder = None

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []


        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []


        memory_samples = {}
        data_generation = []
        seen_des = {}
        self.distill_loss_list = []


        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])
        
        # Log for each task
        current_relations_by_task = []
        all_cur_acc_by_task = []
        test_data_initialize_cur_by_task = []


        for step, (training_data, valid_data, test_data, current_relations, \
            historic_test_data, seen_relations, seen_descriptions) in enumerate(sampler):

            for rel in current_relations:
                ids = self.tokenizer.encode(seen_descriptions[rel][0],
                                    padding='max_length',
                                    truncation=True,
                                    max_length=self.config.max_length)        
                # mask
                mask = np.zeros(self.config.max_length, dtype=np.int32)
                end_index = np.argwhere(np.array(ids) == self.tokenizer.get_vocab()[self.tokenizer.sep_token])[0][0]
                mask[:end_index + 1] = 1 
                if rel not in seen_des:
                    seen_des[rel] = {}
                    seen_des[rel]['ids'] = ids
                    seen_des[rel]['mask'] = mask

            # get representation of seen description
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]

            list_seen_des = []
            for i in range(len(seen_relations)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            # Initialization
            self.moment = Moment(self.config)

            # Train current task
            training_data_initialize = []   

            # if step > 0:
            #     relations = list(set(seen_relations) - set(current_relations))
            #     for rel in relations:
            #         training_data_initialize += memory_samples[rel] 
            #     training_data_initialize += data_generation

            for rel in current_relations:
                training_data_initialize += training_data[rel]

            if self.config.sam_type == 'current':
                self.config.SAM = True
            if self.config.sam_type == 'full' :
                self.config.SAM = True

            if step > 0:
                old_encoder = encoder.get_old_model()
                self.moment.init_moment(encoder, training_data_initialize, is_memory=True)
                self.train_model(encoder, old_encoder, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=True)
            else:
                self.moment.init_moment(encoder, training_data_initialize, is_memory=False)
                self.train_model(encoder, old_encoder, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False)
            if self.config.sam_type == 'current':
                self.config.SAM = False

            # Train memory
            if step > 0:
                old_encoder = encoder.get_old_model()
                relations = list(set(seen_relations) - set(current_relations))
                memory_data_initialize = []
                for rel in relations:
                    memory_data_initialize += memory_samples[rel]
                memory_data_initialize += data_generation
                # augment data:
                data_for_train = training_data_initialize + memory_data_initialize
                if config.mixup:
                    mixup_samples = mixup_data_augmentation(data_for_train)
                    print('Mixup data size: ', len(mixup_samples))
                    self.moment.init_moment_mixup(encoder, mixup_samples, is_memory=True) 
                    self.train_model_mixup(encoder, mixup_samples)

            # Select memory samples
            for rel in current_relations:
                memory_samples[rel], _ = self.select_memory(encoder, training_data[rel])

            # Data gen
            if self.config.use_augment == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))

            # Save the current model state for future tasks
            encoder.set_history()

            # Update proto
            seen_proto = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples[rel])
                seen_proto.append(proto)
            seen_proto = torch.stack(seen_proto, dim=0)

            # get seen relation id
            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            # Eval current task and history task
            test_data_initialize_cur, test_data_initialize_seen = [], []
            for rel in current_relations:
                test_data_initialize_cur += test_data[rel]
            for rel in seen_relations:
                test_data_initialize_seen += historic_test_data[rel]
            
            with torch.no_grad():
                encoder.eval()
                rep_des = []
                for i in range(len(list_seen_des)):
                    sample = {
                        'ids' : torch.tensor([list_seen_des[i]['ids']]).to(self.config.device),
                        'mask' : torch.tensor([list_seen_des[i]['mask']]).to(self.config.device)
                    }
                    hidden = encoder(sample, is_des=True)
                    hidden = hidden.detach().cpu().data
                    rep_des.append(hidden)
                rep_des = torch.cat(rep_des, dim=0)
            encoder.train()

            ac1,ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_cur,rep_des)
            ac2,ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto,seen_relid,test_data_initialize_seen, rep_des)
            
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            print('cur_acc: ', cur_acc)
            print('his_acc: ', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append('{:.4f}'.format(ac1_des))
            total_acc1.append('{:.4f}'.format(ac2_des))
            print('cur_acc des: ', cur_acc1)
            print('his_acc des: ', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append('{:.4f}'.format(ac1_rrf))
            total_acc2.append('{:.4f}'.format(ac2_rrf))
            print('cur_acc rrf: ', cur_acc2)
            print('his_acc rrf: ', total_acc2)

            if self.config.eval_each_task:
                # Eval all current tasks
                cur_acc_by_task = []
                current_relations_by_task.append(current_relations)
                test_data_initialize_cur_by_task.append(test_data_initialize_cur)
                for test_data in test_data_initialize_cur_by_task:
                    cur_acc_by_task.append('{:.4f}'.format(self.eval_encoder_proto(encoder, seen_proto, seen_relid, test_data)))

                all_cur_acc_by_task.append(cur_acc_by_task)
                print('all_cur_acc_by_task: ', all_cur_acc_by_task)
                # logger.info(f"all_cur_acc_by_task: {all_cur_acc_by_task}")


        torch.cuda.empty_cache()
        # save model
        # torch.save(encoder.state_dict(), "./checkpoints/encoder.pth")
        return total_acc_num, total_acc_num1, total_acc_num2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # SAM
    parser.add_argument("--sam", default=False, action='store_true')
    parser.add_argument("--sam_type", default="current", type=str)
    parser.add_argument("--sam_optimizer", default="SAM", type=str)
    parser.add_argument("--rho", default=0.05, type=float)
    parser.add_argument("--decay", default=0, type=float)
    parser.add_argument("--dynamic-rho", action = 'store_true', default=False)
    parser.add_argument("--rho-weight", default=3, type=int)

    # training
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--epoch", default=8, type=int) # 8, 10
    parser.add_argument("--epoch_mem", default=6, type=int) # 6, 10
    parser.add_argument("--eval_each_task", action = 'store_true', default=False)

    # Distill
    parser.add_argument("--distill", default=False, action='store_true')
    parser.add_argument("--distill_type", default="none", type=str)
    parser.add_argument("--distill_top_k", default=5, type=int)

    # Data Augment
    parser.add_argument("--use_augment", default=False, action='store_true')
    parser.add_argument("--mixup", default=False, action='store_true')

    # FewRel
    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=5, type=int)
    parser.add_argument("--lambda_1", default=1.0, type=float)
    parser.add_argument("--lambda_2", default=1.0, type=float)
    parser.add_argument("--lambda_3", default=0.25, type=float)
    parser.add_argument("--temperature", default=0.01, type=float)    
    parser.add_argument("--distance_threshold", default=0.1, type=float)  
    

    # Tacred
    # parser.add_argument("--task_name", default="Tacred", type=str)
    # parser.add_argument("--num_k", default=5, type=int)
    # parser.add_argument("--num_gen", default=5, type=int)
    # parser.add_argument("--lambda_1", default=1, type=float)
    # parser.add_argument("--lambda_2", default=2, type=float)
    # parser.add_argument("--lambda_3", default=0.25, type=float)
    # parser.add_argument("--temperature", default=0.05, type=float)
    # parser.add_argument("--distance_threshold", default=0.3, type=float)

    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    # SAM
    config.sam = args.sam
    config.sam_type = args.sam_type
    config.sam_optimizer = args.sam_optimizer
    config.rho = args.rho
    config.decay = args.decay
    config.dynamic_rho = args.dynamic_rho
    config.rho_weight = args.rho_weight

    # training
    config.batch_size = args.batch_size
    config.epoch = args.epoch
    config.epoch_mem = args.epoch_mem
    config.eval_each_task = args.eval_each_task

    # Distill
    config.distill = args.distill
    config.distill_type = args.distill_type
    config.distill_top_k = args.distill_top_k

    # Hyperparameters
    config.temperature = args.temperature
    config.distance_threshold = args.distance_threshold
    config.lambda_1 = args.lambda_1
    config.lambda_2 = args.lambda_2
    config.lambda_3 = args.lambda_3

    # Data Augment
    config.use_augment = args.use_augment
    config.mixup = args.mixup

    print("SIRIUS Start")
    print(f'dataset: {config.task_name}, {config.num_k}-shot')
    print(f'SAM: {config.sam}')
    print(f'SAM Optimizer: {config.sam_optimizer}')
    print(f'decay: {config.decay}')

    # config 
    print('#############params############')
    print(config.device)
    config.device = torch.device(config.device)
    print(f'Task={config.task_name}, {config.num_k}-shot')
    print(f'Encoding model: {config.model}')
    print(f'pattern={config.pattern}')
    print(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    print('#############params############')

    if config.task_name == 'FewRel':
        config.rel_index = './data/CFRLFewRel/rel_index.npy'
        config.relation_name = './data/CFRLFewRel/relation_name.txt'
        config.relation_description = './data/CFRLFewRel/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLFewRel/CFRLdata_10_100_10_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/train_0.txt'
            config.valid_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/valid_0.txt'
            config.test_data = './data/CFRLFewRel/CFRLdata_10_100_10_10/test_0.txt'
    else:
        config.rel_index = './data/CFRLTacred/rel_index.npy'
        config.relation_name = './data/CFRLTacred/relation_name.txt'
        config.relation_description = './data/CFRLTacred/relation_description.txt'
        if config.num_k == 5:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_5/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_5/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_5/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_5/test_0.txt'
        elif config.num_k == 10:
            config.rel_cluster_label = './data/CFRLTacred/CFRLdata_6_100_5_10/rel_cluster_label_0.npy'
            config.training_data = './data/CFRLTacred/CFRLdata_6_100_5_10/train_0.txt'
            config.valid_data = './data/CFRLTacred/CFRLdata_6_100_5_10/valid_0.txt'
            config.test_data = './data/CFRLTacred/CFRLdata_6_100_5_10/test_0.txt'        

    # seed 
    random.seed(config.seed) 
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)   
    base_seed = config.seed

    acc_list = []
    acc_list1 = []
    aac_list2 = []
    for i in range(config.total_round):
        config.seed = base_seed + i * 100
        print('--------Round ', i)
        config.current_round = i
        print('seed: ', config.seed)
        manager = Manager(config)
        acc, acc1, aac2 = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    std = np.std(accs, axis=0)
    print('----------END')
    print('his_acc mean: ', np.around(ave, 4) * 100)
    print('his_acc std: ', np.around(std, 4) * 100)
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    print('his_acc des mean: ', np.around(ave1, 4) * 100)
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    print('his_acc rrf mean: ', np.around(ave2, 4) * 100)
    print("SIRIUS BERT")
    print(f'SAM: {config.SAM}')
    print(f'SAM_type: {config.sam_type}')
    print(f'SAM Optimizer: {config.sam_optimizer}')
    print(f'decay: {config.decay}')
    