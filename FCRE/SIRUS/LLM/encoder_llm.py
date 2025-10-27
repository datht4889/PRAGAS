import os
import torch
import torch.nn as nn
from llm2vec import LLM2Vec
from torch import Tensor, device, nn
import torch.nn.functional as F

import torch
from transformers import AutoTokenizer, AutoModel
from peft import LoraConfig, get_peft_model
from typing import Any, Dict, List, Optional, Tuple, Union

def batch_to_device(batch, target_device: device):

    for key in batch:
        if isinstance(batch[key], Tensor):
            batch[key] = batch[key].to(target_device)
    return batch

class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config

    def forward(self, inputs, is_des = False, is_distill = False, top_k=10): # (b, max_length)
        batch_size = len(inputs)
        features = self.encoder.tokenize(inputs)
        features = batch_to_device(features, self.config.device)

        # Get model outputs with attention scores
        outputs = self.encoder.model(
            input_ids=features['input_ids'],
            attention_mask=features['attention_mask'],
            output_attentions=True if is_distill else None
        )

        # Get last hidden states
        last_hidden_state = outputs.last_hidden_state
        attention_scores = outputs.attentions
        embeddings = self.encoder.get_pooling(features, last_hidden_state)  # Shape: [batch_size, hidden_size]
        assert embeddings.shape[0] == batch_size, f"{embeddings.shape[0]} != {batch_size}, {embeddings.shape}, {last_hidden_state.shape}"
        if is_distill:
            # Get the last layer attention scores (similar to EncodingModel)
            attentions_layer = attention_scores[-1]  # choose last layer
            
            # Column-wise sum over queries (dim=2). Result: (B, H, S)
            col_sum_per_head = attentions_layer.sum(dim=2)
            
            # Aggregate across heads -> (B, S)
            token_scores = col_sum_per_head.mean(dim=1)  # (B, S)
            
            # Mask out padding tokens
            attention_mask = features['attention_mask']
            mask = attention_mask.to(token_scores.dtype)
            token_scores = token_scores * mask
            
            # Normalize to probabilities per example
            token_probs = token_scores / (token_scores.sum(dim=1, keepdim=True) + 1e-12)
            
            # Choose safe k (don't request more than seq length)
            S = token_probs.size(1)
            k = top_k if S > top_k else S
            
            # top-k indices and scores per example
            topk_scores, topk_indices = torch.topk(token_probs, k=k, dim=1)
            
            # Prepare indices for gathering hidden states
            B, _, H = last_hidden_state.size()
            topk_hidden_indices = topk_indices.unsqueeze(-1).expand(-1, -1, H)  # (B, k, H)
            return embeddings, last_hidden_state, topk_hidden_indices



        return embeddings
    
    def set_history(self):
        """Store the current model state for knowledge distillation"""
        # Clean up old_model before saving
        if hasattr(self, 'old_model'):
            self.cleanup_old_model()
        
        # Get state_dict and filter out old_model keys
        main_state_dict = {k: v for k, v in self.state_dict().items() 
                        if not k.startswith('old_model.')}
        
        self.history = {
            "state_dict": main_state_dict,
        }
    
    def get_old_model(self):
        if self.history is None:
            raise ValueError("No history saved. Call set_history() before training on new tasks.")
        
        # Create an instance of the same class (e.g., EncodingModel_Llama3)
        self.old_model = self.__class__(self.config)
        self.old_model.load_state_dict(self.history["state_dict"])
        self.old_model.eval()
        self.old_model.to(self.config.device)
        
        # Freeze the old model to save memory
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        return self.old_model
    
    def cleanup_old_model(self):
        """Clean up the old model to free memory"""
        if hasattr(self, 'old_model') and self.old_model is not None:
            del self.old_model
            self.old_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

class EncodingModel_Llama2(EncodingModel):
    def __init__(self, config):
        EncodingModel.__init__(self, config)
        self.config = config

        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Llama-2-7b-chat-hf-mntp-supervised",
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=256,
            token=os.getenv('HUGGINGFACE_API_KEY')
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model


class EncodingModel_Llama3(EncodingModel):
    def __init__(self, config):
        EncodingModel.__init__(self, config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=256,
            token=os.getenv('HUGGINGFACE_API_KEY')
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model


class EncodingModel_Mistral(EncodingModel):
    def __init__(self, config):
        EncodingModel.__init__(self, config)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
        )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            attn_implementation="eager",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=512,
            skip_instruction = False,
            token=os.getenv('HUGGINGFACE_API_KEY')
            
        )
        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
            
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        if lora_modules is None and model.config.__class__.__name__ in [
            "LlamaConfig",
            "MistralConfig",
            "GemmaConfig",
            "Qwen2Config",
        ]:
            lora_modules = [
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
        elif lora_modules is None:
            raise ValueError("lora_modules must be specified for this model.")

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

class EncodingModel_BGE(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
        self.model = AutoModel.from_pretrained('BAAI/bge-en-icl', 
                                               attn_implementation="eager",
                                               device_map="cuda" if torch.cuda.is_available() else "cpu", 
                                               trust_remote_code=True, 
                                               token=os.getenv('HUGGINGFACE_API_KEY'),
                                               torch_dtype = torch.bfloat16)
        self.model = self.initialize_peft(
            self.model,
        )
    
    def last_token_pool(self, last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
                
    def initialize_peft(
        self,
        model,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_modules: Optional[List[str]] = None,
    ):
        
        lora_modules = [
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
       

        config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=None,
        )

        model = get_peft_model(model, config)
        print(f"Model's Lora trainable parameters:")
        model.print_trainable_parameters()
        return model

    def forward(self, input_texts, is_des = False, is_distill = False, top_k=10): # (b, max_length)
        batch_size = len(input_texts)
        input_texts = self.tokenizer(input_texts, max_length=256, padding=True, truncation=True, return_tensors='pt').to(self.config.device)

        outputs = self.model(**input_texts, output_attentions=True if is_distill else None)
        
        last_hidden_state = outputs.last_hidden_state
        query_embeddings = self.last_token_pool(last_hidden_state, input_texts['attention_mask'])
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)

        if is_distill:
            attention_scores = outputs.attentions
            
            # Get the last layer attention scores
            attentions_layer = attention_scores[-1]  # choose last layer
            
            # Column-wise sum over queries (dim=2). Result: (B, H, S)
            col_sum_per_head = attentions_layer.sum(dim=2)
            
            # Aggregate across heads -> (B, S)
            token_scores = col_sum_per_head.mean(dim=1)  # (B, S)
            
            # Mask out padding tokens
            attention_mask = input_texts['attention_mask']
            mask = attention_mask.to(token_scores.dtype)
            token_scores = token_scores * mask
            
            # Normalize to probabilities per example
            token_probs = token_scores / (token_scores.sum(dim=1, keepdim=True) + 1e-12)
            
            # Choose safe k (don't request more than seq length)
            S = token_probs.size(1)
            k = top_k if S > top_k else S
            
            # top-k indices and scores per example
            topk_scores, topk_indices = torch.topk(token_probs, k=k, dim=1)
            
            # Prepare indices for gathering hidden states
            B, _, H = last_hidden_state.size()
            topk_hidden_indices = topk_indices.unsqueeze(-1).expand(-1, -1, H)  # (B, k, H)
            
            return query_embeddings, last_hidden_state, topk_hidden_indices

        return query_embeddings

    def set_history(self):
        """Store the current model state for knowledge distillation"""
        # Clean up old_model before saving
        if hasattr(self, 'old_model'):
            self.cleanup_old_model()
        
        # Get state_dict and filter out old_model keys
        main_state_dict = {k: v for k, v in self.state_dict().items() 
                        if not k.startswith('old_model.')}
        
        self.history = {
            "state_dict": main_state_dict,
        }

    def get_old_model(self):
        if self.history is None:
            raise ValueError("No history saved. Call set_history() before training on new tasks.")
        
        self.old_model = self.__class__(self.config)
        self.old_model.load_state_dict(self.history["state_dict"], strict=False)
        self.old_model.to(self.config.device)
        self.old_model.eval()
        
        # Freeze the old model to save memory
        for param in self.old_model.parameters():
            param.requires_grad = False
        
        return self.old_model

    def cleanup_old_model(self):
        """Clean up the old model to free memory"""
        if hasattr(self, 'old_model') and self.old_model is not None:
            del self.old_model
            self.old_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        

