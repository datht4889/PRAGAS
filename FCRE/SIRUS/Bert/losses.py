from __future__ import annotations
from typing import Any, Iterable
import torch
from torch import Tensor, nn
import numpy as np 
from enum import Enum
from torch.nn import functional as F


def _convert_to_tensor(a: list | np.ndarray | Tensor):
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a

def _convert_to_batch(a: Tensor):
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor):
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))

class TripletDistanceMetric(Enum):
    """The metric for the triplet loss"""

    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)


class TripletLoss(nn.Module):
    def __init__(
        # self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0
        self, distance_metric=TripletDistanceMetric.COSINE, triplet_margin: float = 1.0

    ) -> None:
        """
        This class implements triplet loss. Given a triplet of (anchor, positive, negative),
        the loss minimizes the distance between anchor and positive while it maximizes the distance
        between anchor and negative. It compute the following loss function:

        ``loss = max(||anchor - positive|| - ||anchor - negative|| + margin, 0)``.

        Margin is an important hyperparameter and needs to be tuned respectively.

        Args:
            model: SentenceTransformerModel
            distance_metric: Function to compute distance between two
                embeddings. The class TripletDistanceMetric contains
                common distance metrices that can be used.
            triplet_margin: The negative should be at least this much
                further away from the anchor than the positive.

        """
        super().__init__()
        # self.model = model
        self.distance_metric = distance_metric
        self.triplet_margin = triplet_margin

    def forward(self, rep_anchor, rep_pos, rep_neg) -> Tensor:

        distance_pos = self.distance_metric(rep_anchor, rep_pos)
        distance_neg = self.distance_metric(rep_anchor, rep_neg)
    
        losses = F.relu(distance_pos - distance_neg + self.triplet_margin)
       
        # losses = torch.where(losses == self.triplet_margin, torch.tensor(0.0, device=losses.device), losses)

        return losses.mean()

### Mixup Loss ###
class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it requires more
              training time.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    #     """Returns token_embeddings, cls_token"""
    #     trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}


# khi load input vi du la 1 batch thi input dau vao phai la 1 cai dict giua input dau vao description tuong ung 

#     The embeddings for the anchor sentences are stored in embeddings_a, and the embeddings for the positive sentences are concatenated into embeddings_b.
# The similarity scores between embeddings_a and embeddings_b are computed using the provided similarity function and then scaled.
    def forward(self, embeddings_a, embeddings_b, labels) -> Tensor:
        # reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # Example a[i] should match with b[i]

        #  print(scores.shape) b*b
        # range_labels.shape = b
        # range_labels = torch.arange(0, scores.size(0), device=scores.device)
        
        return self.cross_entropy_loss(scores, labels)



    
    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}

class NegativeCosSimLoss(nn.Module):
    def __init__(self, temperature: float = 2):
        super().__init__()
        self.temperature = temperature
        

    def forward(self, ebd1, ebd2):
        
        # Normalize the embeddings to ensure they are unit vectors
        batch1 = F.normalize(ebd1, p=2, dim=-1)
        batch2 = F.normalize(ebd2, p=2, dim=-1)
        # add noise to the embeddings
        batch1 = batch1 + torch.randn_like(batch1) * 1e-5
        batch2 = batch2 + torch.randn_like(batch2) * 1e-5
        
        # Compute the cosine similarity for each corresponding pair in the batches
        cos_sim = torch.sum(batch1 * batch2, dim=-1) / self.temperature
        
        # Negate the cosine similarity to maximize the distance
        # loss = - torch.log(1 + exp(cos_sim))
        # loss = - torch.log(1 + torch.exp(cos_sim)).mean()
        exp_sim = torch.exp(cos_sim)
        loss = torch.log((1 + exp_sim)).mean()
        return loss

### Distillation Loss ###

class RKdAngle(nn.Module):
    def forward(self, student, teacher):
        # N x C
        # N x N x C

        with torch.no_grad():
            td = (teacher.unsqueeze(0) - teacher.unsqueeze(1))
            norm_td = F.normalize(td, p=2, dim=2)
            t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)

        sd = (student.unsqueeze(0) - student.unsqueeze(1))
        norm_sd = F.normalize(sd, p=2, dim=2)
        s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)

        loss = F.smooth_l1_loss(s_angle, t_angle, reduction='elementwise_mean')
        return loss

class RkdDistance(nn.Module):
    def pdist(self, e, squared=False, eps=1e-12):
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)

        if not squared:
            res = res.sqrt()

        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res
    
    def forward(self, student, teacher):
        with torch.no_grad():
            t_d = self.pdist(teacher, squared=False)
            mean_td = t_d[t_d>0].mean()
            t_d = t_d / mean_td

        d = self.pdist(student, squared=False)
        mean_d = d[d>0].mean()
        d = d / mean_d

        loss = F.smooth_l1_loss(d, t_d, reduction='elementwise_mean')
        return loss

class Distiller(nn.Module):
    def remap_logit(self, 
                    logits_student: torch.Tensor,  # [B, N] 
                    logits_teacher: torch.Tensor,  # [B, N]
                    seen_classes: list,            # [N]
                    total_class: int               # total_class ≥ N
    )-> torch.Tensor:
        """
        Remap logits_student and logits_teacher to total class softmax.
        """
        new_logits_student = torch.zeros(logits_student.shape[0], total_class, device=logits_student.device) # [B, total_class]
        new_logits_teacher = torch.zeros(logits_teacher.shape[0], total_class, device=logits_teacher.device) # [B, total_class]
        assert len(seen_classes) == logits_student.shape[1], (
            f"seen_classes should be the same as the number of classes in the teacher. "
            f"{len(seen_classes)} != {logits_student.shape[1]}. Seen classes: {seen_classes}."
        )
        for b in range(logits_student.shape[0]):
            for idx in range(logits_student.shape[1]):
                new_logits_student[b][seen_classes[idx]] = logits_student[b][idx]
                new_logits_teacher[b][seen_classes[idx]] = logits_teacher[b][idx]

        return new_logits_student, new_logits_teacher    


class RKD(Distiller):
    def __init__(self, temperature: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def rkd_angle(
                self, 
                student, # [B, K, N]
                teacher  # [B, K, N]
                ):
        B, K, N = student.shape
        total_loss = 0.0
        
        # Process each batch separately
        for b in range(B):
            student_b = student[b]  # [K, N]
            teacher_b = teacher[b]  # [K, N]
            
            # Compute teacher angles for this batch
            with torch.no_grad():
                td = (teacher_b.unsqueeze(0) - teacher_b.unsqueeze(1))  # [K, K, N]
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
            
            # Compute student angles for this batch
            sd = (student_b.unsqueeze(0) - student_b.unsqueeze(1))  # [K, K, N]
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
            
            batch_loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
            total_loss += batch_loss
        
        return total_loss / B

    def rkd_distance(
            self, 
            student, # [B, K, N]
            teacher  # [B, K, N]
            ):
        B, K, N = student.shape
        total_loss = 0.0
        
        # Process each batch separately
        for b in range(B):
            student_b = student[b]  # [K, N]
            teacher_b = teacher[b]  # [K, N]
            
            # Compute teacher distances for this batch
            with torch.no_grad():
                t_d = self._pdist(teacher_b, squared=False)
                mean_td = t_d[t_d > 0].mean()
                t_d = t_d / mean_td
            
            # Compute student distances for this batch
            d = self._pdist(student_b, squared=False)
            mean_d = d[d > 0].mean()
            d = d / mean_d
            
            batch_loss = F.smooth_l1_loss(d, t_d, reduction='mean')
            total_loss += batch_loss
        
        return total_loss / B

    def _pdist(self, e, squared=False, eps=1e-12):
        # e: [K, N]
        e_square = e.pow(2).sum(dim=1)
        prod = e @ e.t()
        res = (e_square.unsqueeze(1) + e_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
        if not squared:
            res = res.sqrt()
        res = res.clone()
        res[range(len(e)), range(len(e))] = 0
        return res

    def forward(
        self,
        logits_student: torch.Tensor,  # [B, K, N]
        logits_teacher: torch.Tensor,  # [B, K, N]
        *args,
        **kwargs
    ):
        # convert to cuda
        logits_student = logits_student.to(self.device)
        logits_teacher = logits_teacher.to(self.device)
        # labels = labels.to(self.device)

        loss = self.rkd_angle(logits_student, logits_teacher) + self.rkd_distance(logits_student, logits_teacher)
        return loss

class KLDivAndAngleLoss(Distiller):
    def __init__(self, temperature: float = 1.0, device: str = 'cpu'):
        super().__init__()
        self.temperature = temperature
        self.device = device

    def rkd_angle(
                self, 
                student, # [B, K, N]
                teacher  # [B, K, N]
                ):
        B, K, N = student.shape
        total_loss = 0.0
        
        # Process each batch separately
        for b in range(B):
            student_b = student[b]  # [K, N]
            teacher_b = teacher[b]  # [K, N]
            
            # Compute teacher angles for this batch
            with torch.no_grad():
                td = (teacher_b.unsqueeze(0) - teacher_b.unsqueeze(1))  # [K, K, N]
                norm_td = F.normalize(td, p=2, dim=2)
                t_angle = torch.bmm(norm_td, norm_td.transpose(1, 2)).view(-1)
            
            # Compute student angles for this batch
            sd = (student_b.unsqueeze(0) - student_b.unsqueeze(1))  # [K, K, N]
            norm_sd = F.normalize(sd, p=2, dim=2)
            s_angle = torch.bmm(norm_sd, norm_sd.transpose(1, 2)).view(-1)
            
            batch_loss = F.smooth_l1_loss(s_angle, t_angle, reduction='mean')
            total_loss += batch_loss
        
        return total_loss / B

    def forward(
        self,
        logits_student: torch.Tensor,  # [B, K, N]
        logits_teacher: torch.Tensor,  # [B, K, N]
        labels=None,  # Fixed: added labels parameter with default None
        *args,
        **kwargs
    ):
        # Convert to cuda
        logits_student = logits_student.to(self.device)
        logits_teacher = logits_teacher.to(self.device)
        if labels is not None:
            labels = labels.to(self.device)

        B, K, N = logits_student.shape
        
        # Normalize for cosine similarity stability
        logits_student = F.normalize(logits_student, p=2, dim=-1)
        logits_teacher = F.normalize(logits_teacher, p=2, dim=-1)

        # Process KL divergence loss for each batch and token
        kl_loss = 0.0
        for b in range(B):
            for k in range(K):
                # Get logits for current batch and token
                student_logits = logits_student[b, k]  # [N]
                teacher_logits = logits_teacher[b, k]  # [N]
                
                # Probability distributions
                pred_student = F.log_softmax(student_logits / self.temperature, dim=0)
                pred_teacher = F.softmax(teacher_logits / self.temperature, dim=0)
                
                # KL Divergence Loss for this token
                kl_loss += F.kl_div(pred_student, pred_teacher, reduction='sum')
        
        # Average over batch and tokens
        kl_loss = kl_loss / (B * K)

        # Angle Loss (already handles batch processing correctly)
        angle_loss = self.rkd_angle(logits_student, logits_teacher)

        return kl_loss + angle_loss
