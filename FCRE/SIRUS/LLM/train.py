import argparse
import torch
import random
import sys
import copy
import logging
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


from sampler_llm import data_sampler_CFRL
from data_loader import get_data_loader_BERT
from utils import Moment, gen_data
from encoder_llm import EncodingModel_Llama2, EncodingModel_Llama3, EncodingModel_Mistral, EncodingModel_BGE 

from transformers import AutoTokenizer, AutoModel, AutoConfig

# import wandb

from transformers import BertTokenizer
from losses import *

class Manager(object):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        
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
    

    def get_memory_proto(self, encoder, dataset, is_llm = False):
        '''
        only for one relation data
        '''
        data_loader = get_data_loader_BERT(config, dataset, shuffle=False, \
            drop_last=False,  batch_size=1) 
        features = []
        encoder.eval()
        for step, (instance, label, idx) in enumerate(data_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            if is_llm:
                hidden = encoder(instance['input'])
            else:
                hidden = encoder(instance) 
            fea = hidden.detach().cpu().data # (1, H)
            features.append(fea)    
        features = torch.cat(features, dim=0) # (M, H)
        proto = features.mean(0)

        return proto, features   

    def select_memory(self, encoder, dataset, is_llm = False):
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
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            if is_llm:
                hidden = encoder(instance['input'])
            else:
                hidden = encoder(instance) 

            fea = hidden.detach().cpu().float().numpy()  # Convert to float32 and then to numpy array
            # fea = hidden.detach().cpu().data # (1, H)
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
        features = torch.from_numpy(features) # (N, H) tensor
        rel_proto = features.mean(0) # (H)

        return mem_set, mem_feas

    def get_cluster_and_centroids(self, embeddings):
        clustering_model = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=args.distance_threshold)

        clusters = clustering_model.fit_predict(embeddings.to(dtype=torch.float32).cpu().numpy())

        centroids = {}
        for cluster_id in np.unique(clusters):
            if cluster_id not in centroids:
                cluster_embeddings = embeddings[clusters == cluster_id]

                centroid = torch.mean(cluster_embeddings, dim=0)

                centroids[cluster_id] = centroid

        return clusters, centroids

    def train_model(self, encoder, old_encoder, training_data, seen_des, seen_relations, list_seen_des, is_memory=False):
        data_loader = get_data_loader_BERT(self.config, training_data, shuffle=True)
        optimizer = optim.AdamW(params=encoder.parameters(), lr=self.config.lr)

        if self.config.sam:
            # base_optimizer = optim.AdamW
            # optimizer = SAM(params=encoder.parameters(), base_optimizer=base_optimizer, rho=self.config.rho, adaptive=True, lr=self.config.lr)
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
                    if isinstance(instance[k], list):
                        continue
                    else:
                        instance[k] = instance[k].to(self.config.device)
                batch_instance = {'input': []}
                batch_instance['input'] = [seen_des[self.id2rel[label.item()]]['input'] for label in labels]
                
                rep_seen_des = []
                with torch.no_grad(): 
                    for i2 in range(len(list_seen_des)):
                        sample = {
                            'input' : list_seen_des[i2]['input']
                        }
                        hidden = encoder(sample['input'])
                        # hidden = hidden.detach().cpu()
                        rep_seen_des.append(hidden)
                    rep_seen_des = torch.cat(rep_seen_des, dim=0)
                    rep_seen_des = rep_seen_des.detach().cpu()

                hidden, outputs_words, topk_hidden_indices = encoder(instance['input'], is_distill=True, top_k=self.config.distill_top_k)
                rep_des = encoder(batch_instance['input']) # b, di

                clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)

                flag = 0
                if len(clusters) == max(clusters) + 1:
                    flag = 1

                relationid2_clustercentroids = {}
                for index, rel in enumerate(seen_relations):
                    relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                relation_2_cluster = {}
                for i1 in range(len(seen_relations)):
                    relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]
                
                loss2 = self.llm_moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=args.temperature ,relation_2_cluster=relation_2_cluster)  # Recompute loss2

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
                        logger.debug(f"RuntimeError in top-k selection: {e}")
                        top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                    nearest_cluster_centroids.append(top2_centroids)

                nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)
                if flag == 0:
                    loss1 = self.llm_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)

                    loss = args.lambda_1*loss1 + args.lambda_2*loss2 + args.lambda_3*loss3

                else:
                    loss1 = self.llm_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                    loss = args.lambda_1*loss1 + args.lambda_2*loss2 

                if old_encoder is not None and self.config.distill and self.config.distill_type != 'none':
                    old_hidden, old_outputs_words, old_topk_hidden_indices = old_encoder(instance['input'], is_distill=True, top_k=self.config.distill_top_k)
                    old_topk_hidden = torch.gather(old_outputs_words, dim=1, index=old_topk_hidden_indices)  # (B, k, H)
                    topk_hidden = torch.gather(outputs_words, dim=1, index=topk_hidden_indices)  # (B, k, H)
                    if self.config.distill_type in ['RKD', 'KLDivAndAngleLoss']:
                        distill_loss = distill_loss_fn(topk_hidden, old_topk_hidden)
                        # loss = loss + distill_loss * self.config.distill_loss_weight
                        self.distill_loss_list.append(distill_loss.item())
                    else:
                        raise NotImplementedError("Distill Loss {} not implemented".format(self.config.distill_type)) 
    
                if not self.config.sam:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                else:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.first_step(zero_grad=True, rho=self.config.rho)

                    hidden = encoder(instance['input']) # b, dim
                    rep_des = encoder(batch_instance['input']) # b, di

                    clusters, clusters_centroids = self.get_cluster_and_centroids(rep_seen_des)

                    flag = 0
                    if len(clusters) == max(clusters) + 1:
                        flag = 1

                    relationid2_clustercentroids = {}
                    for index, rel in enumerate(seen_relations):
                        relationid2_clustercentroids[self.rel2id[rel]] = clusters_centroids[clusters[index]]

                    relation_2_cluster = {}
                    for i1 in range(len(seen_relations)):
                        relation_2_cluster[self.rel2id[seen_relations[i1]]] = clusters[i1]

                    loss2 = self.llm_moment.mutual_information_loss_cluster(hidden, rep_des, labels, temperature=args.temperature ,relation_2_cluster=relation_2_cluster)  # Recompute loss2

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
                            logger.debug(f"RuntimeError in top-k selection: {e}")
                            top2_centroids = relationid2_clustercentroids[labels[torch.argmax(cos_similarities).item()].item()]

                        nearest_cluster_centroids.append(top2_centroids)

                    nearest_cluster_centroids = torch.stack(nearest_cluster_centroids, dim = 0).to(self.config.device)
                    if flag == 0:
                        loss1 = self.llm_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                        loss3 = triplet(hidden, rep_des,  cluster_centroids) + triplet(hidden, cluster_centroids, nearest_cluster_centroids)

                        loss = args.lambda_1*loss1 + args.lambda_2*loss2 + args.lambda_3*loss3

                    else:
                        loss1 = self.llm_moment.contrastive_loss(hidden, labels, is_memory, des =rep_des, relation_2_cluster = relation_2_cluster)

                        loss = args.lambda_1*loss1 + args.lambda_2*loss2  

                    if old_encoder is not None and self.config.distill and self.config.distill_type != 'none':
                        old_hidden, old_outputs_words, old_topk_hidden_indices = old_encoder(instance['input'], is_distill=True, top_k=self.config.distill_top_k)
                        old_topk_hidden = torch.gather(old_outputs_words, dim=1, index=old_topk_hidden_indices)  # (B, k, H)
                        topk_hidden = torch.gather(outputs_words, dim=1, index=topk_hidden_indices)  # (B, k, H)
                        if self.config.distill_type in ['RKD', 'KLDivAndAngleLoss']:
                            distill_loss = distill_loss_fn(topk_hidden, old_topk_hidden)
                            # loss = loss + distill_loss * self.config.distill_loss_weight
                        else:
                            raise NotImplementedError("Distill Loss {} not implemented".format(self.config.distill_type))
        
                    loss.backward()
                    optimizer.second_step()

                # update moment
                if is_memory:
                    self.llm_moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=True)
                else:
                    self.llm_moment.update_des(ind, hidden.detach().cpu().data, rep_des.detach().cpu().data, is_memory=False)
                # print
                if is_memory:
                    sys.stdout.write('MemoryTrain:  epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                else:
                    sys.stdout.write('CurrentTrain: epoch {0:2}, batch {1:5} | loss: {2:2.7f}'.format(i, batch_num, loss.item()) + '\r')
                sys.stdout.flush() 
        logger.info('')             
                   

    def eval_encoder_proto(self, encoder, seen_proto, seen_relid, test_data):
        batch_size = 16
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)
        
        corrects = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
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
        logger.info('')
        return corrects / total
    def eval_encoder_proto_des(self, encoder, seen_proto, seen_relid, test_data, rep_des):
    
        batch_size = 4
        test_loader = get_data_loader_BERT(self.config, test_data, False, False, batch_size)

        corrects = 0.0
        corrects1 = 0.0
        corrects2 = 0.0
        total = 0.0
        encoder.eval()
        for batch_num, (instance, label, _) in enumerate(test_loader):
            for k in instance.keys():
                if isinstance(instance[k], list):
                    continue
                else:
                    instance[k] = instance[k].to(self.config.device)
            # hidden = encoder(instance)

            hidden = encoder(instance['input'])
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
        logger.info('')
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

        logger.info('prepared data!')
        self.id2rel = sampler.id2rel
        self.rel2id = sampler.rel2id
        self.r2desc = self._read_description(self.config.relation_description)

        if self.config.backbone == 'llama3':
            encoder = EncodingModel_Llama3(self.config)
        elif self.config.backbone == 'llama2':
            encoder = EncodingModel_Llama2(self.config)
        elif self.config.backbone == 'mistral':
            encoder = EncodingModel_Mistral(self.config)
        elif self.config.backbone == 'bge':
            encoder = EncodingModel_BGE(self.config)
        old_encoder = None

        # step is continual task number
        cur_acc, total_acc = [], []
        cur_acc1, total_acc1 = [], []
        cur_acc2, total_acc2 = [], []


        cur_acc_num, total_acc_num = [], []
        cur_acc_num1, total_acc_num1 = [], []
        cur_acc_num2, total_acc_num2 = [], []


        data_generation = []
        seen_des = {}

        memory_samples_llm = {}
        self.unused_tokens = ['[unused0]', '[unused1]', '[unused2]', '[unused3]']
        self.unused_token = '[unused0]'
        self.tokenizer = BertTokenizer.from_pretrained(self.config.bert_path, \
            additional_special_tokens=[self.unused_token])
        
        # self.stella_tokenizer = AutoTokenizer.from_pretrained(self.config.llm_path, trust_remote_code=True)



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
                    seen_des[rel]['input'] = seen_descriptions[rel]

            seen_relid = []
            for rel in seen_relations:
                seen_relid.append(self.rel2id[rel])

            seen_des_by_id = {}
            for rel in seen_relations:
                seen_des_by_id[self.rel2id[rel]] = seen_des[rel]

            list_seen_des = []
            for i in range(len(seen_relations)):
                list_seen_des.append(seen_des_by_id[seen_relid[i]])

            # train_llm first

            # Initialization
            self.llm_moment = Moment(self.config)

            # Train current task
            training_data_initialize = []

            if step > 0:
                relations = list(set(seen_relations) - set(current_relations))
                for rel in relations:
                    training_data_initialize += memory_samples_llm[rel] 
                training_data_initialize += data_generation 
            
            for rel in current_relations:
                training_data_initialize += training_data[rel]

            if self.config.sam_type == 'current':
                self.config.SAM = True
            if self.config.sam_type == 'full' :
                self.config.SAM = True       

            if step > 0:
                old_encoder = encoder.get_old_model()
            self.llm_moment.init_moment(encoder, training_data_initialize, seen_des, self.id2rel, is_memory=False, is_llm = True)
            self.train_model(encoder, old_encoder, training_data_initialize, seen_des, seen_relations, list_seen_des, is_memory=False)
            if self.config.sam_type == 'current':
                self.config.SAM = False

            # Select memory samples
            for rel in current_relations:
                memory_samples_llm[rel], _ = self.select_memory(encoder, training_data[rel], is_llm = True)

            # Data gen
            if self.config.use_augment == 1:
                gen_text = []
                for rel in current_relations:
                    for sample in memory_samples_llm[rel]:
                        sample_text = self._get_sample_text(self.config.training_data, sample['index'])
                        gen_samples = gen_data(self.r2desc, self.rel2id, sample_text, self.config.num_gen, self.config.gpt_temp, self.config.key)
                        gen_text += gen_samples
                for sample in gen_text:
                    data_generation.append(sampler.tokenize(sample))

            # Save the current model state for future tasks
            encoder.set_history()

            # Update proto
            seen_proto_llm = []  
            for rel in seen_relations:
                proto, _ = self.get_memory_proto(encoder, memory_samples_llm[rel], is_llm = True)
                seen_proto_llm.append(proto)
            seen_proto_llm = torch.stack(seen_proto_llm, dim=0)

            # end training phrase 

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

            rep_des = []

            with torch.no_grad():
                encoder.eval()
                for i in range(len(list_seen_des)):
                    sample = {
                            'input' : [list_seen_des[i]['input']]
                        }
                    hidden = encoder(sample['input'])
                    hidden = hidden.detach().cpu().data
                    rep_des.append(hidden)
                rep_des = torch.cat(rep_des, dim=0)
            encoder.train()

            ac1,ac1_des, ac1_rrf = self.eval_encoder_proto_des(encoder,seen_proto_llm,seen_relid,test_data_initialize_cur,rep_des)
            ac2,ac2_des, ac2_rrf = self.eval_encoder_proto_des(encoder,seen_proto_llm,seen_relid,test_data_initialize_seen, rep_des)
            
            cur_acc_num.append(ac1)
            total_acc_num.append(ac2)
            cur_acc.append('{:.4f}'.format(ac1))
            total_acc.append('{:.4f}'.format(ac2))
            logger.info('cur_acc: %s', cur_acc)
            logger.info('his_acc: %s', total_acc)

            cur_acc_num1.append(ac1_des)
            total_acc_num1.append(ac2_des)
            cur_acc1.append('{:.4f}'.format(ac1_des))
            total_acc1.append('{:.4f}'.format(ac2_des))
            logger.info('cur_acc des: %s', cur_acc1)
            logger.info('his_acc des: %s', total_acc1)

            cur_acc_num2.append(ac1_rrf)
            total_acc_num2.append(ac2_rrf)
            cur_acc2.append('{:.4f}'.format(ac1_rrf))
            total_acc2.append('{:.4f}'.format(ac2_rrf))
            logger.info('cur_acc rrf: %s', cur_acc2)
            logger.info('his_acc rrf: %s', total_acc2)


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

    # Distill
    parser.add_argument("--distill", default=False, action='store_true')
    parser.add_argument("--distill_type", default="none", type=str)
    parser.add_argument("--distill_top_k", default=5, type=int)

    # Data Augment
    parser.add_argument("--use_augment", default=False, action='store_true')

    parser.add_argument("--task_name", default="FewRel", type=str)
    parser.add_argument("--num_k", default=5, type=int)
    parser.add_argument("--num_gen", default=2, type=int)
    parser.add_argument("--lambda_1", default=1, type=float)
    parser.add_argument("--lambda_2", default=2, type=float)
    parser.add_argument("--lambda_3", default=0.5, type=float)
    parser.add_argument("--temperature", default=0.05, type=float)
    parser.add_argument("--distance_threshold", default=0.3, type=float)
    parser.add_argument("--model", default='bert', type=str) 
    parser.add_argument("--backbone", default='llama3', type=str) # llama3, llama2, bge, mistral

    args = parser.parse_args()
    config = Config('config.ini')
    config.task_name = args.task_name
    config.num_k = args.num_k
    config.num_gen = args.num_gen
    config.lambda_1 = args.lambda_1
    config.lambda_2 = args.lambda_2
    config.lambda_3 = args.lambda_3
    config.temperature = args.temperature
    config.distance_threshold = args.distance_threshold
    config.model = args.model
    config.backbone = args.backbone

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

    # Distill
    config.distill = args.distill
    config.distill_type = args.distill_type
    config.distill_top_k = args.distill_top_k

    # Data Augment
    config.use_augment = args.use_augment


    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(f'SIRUS_backbone_{args.task_name}_{args.backbone}_{args.sam_optimizer}.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)

    # config 
    logger.info('#############params############')
    logger.info(config.device)
    config.device = torch.device(config.device)
    logger.info(f'Task={config.task_name}, {config.num_k}-shot')
    logger.info(f'Encoding model: {config.backbone}')
    logger.info(f'pattern={config.pattern}')
    logger.info(f'mem={config.memory_size}, margin={config.margin}, gen={config.gen}, gen_num={config.num_gen}')
    logger.info('#############params############')

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
        logger.info('--------Round %s', i)
        config.current_round = i
        logger.info('seed: %s', config.seed)
        manager = Manager(config)
        acc, acc1, aac2 = manager.train()
        acc_list.append(acc)
        acc_list1.append(acc1)
        aac_list2.append(aac2)
        torch.cuda.empty_cache()
    
    accs = np.array(acc_list)
    ave = np.mean(accs, axis=0)
    std = np.std(accs, axis=0)
    logger.info('----------END')
    logger.info('his_acc mean: %s', np.around(ave, 4)*100)
    logger.info('his_acc std: %s', np.around(std, 4)*100)
    accs1 = np.array(acc_list1)
    ave1 = np.mean(accs1, axis=0)
    std1 = np.std(accs1, axis=0)
    logger.info('his_acc des mean: %s', np.around(ave1, 4)*100)
    logger.info('his_acc des std: %s', np.around(std1, 4)*100)
    accs2 = np.array(aac_list2)
    ave2 = np.mean(accs2, axis=0)
    std2 = np.std(accs2, axis=0)
    logger.info('his_acc rrf mean: %s', np.around(ave2, 4)*100)
    logger.info('his_acc rrf std: %s', np.around(std2, 4)*100)

# TOKENIZERS_PARALLELISM=true CUDA_VISIBLE_DEVICES=0 python train.py --task_name Tacred \
#     --num_k 5 \
#     --num_gen 5 \
#     --lambda_1 1 \
#     --lambda_2 2 \
#     --lambda_3 0.25 \
#     --temperature 0.05 \
#     --distance_threshold 0.3 \
#     --decay 0 \
#     --sam \
#     --sam_optimizer ASAM \
#     --rho 0.05 \
#     --dynamic-rho \
#     --rho-weight 4 \
#     --batch_size 8 \
#     --epoch 10 \
#     --epoch_mem 10 \
#     --distill \
#     --distill_type RKD \
#     --distill_top_k 10 \
#     --backbone bge \
#     --use_augment