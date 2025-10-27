import torch
import torch.nn as nn
import numpy as np
from transformers import BertModel
from transformers import RobertaModel
from transformers import BertForMaskedLM
class EncodingModel(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        if config.model == 'bert':
            self.encoder = BertModel.from_pretrained(config.bert_path).to(config.device)
            self.lm_head = BertForMaskedLM.from_pretrained(config.bert_path).to(config.device).cls
        elif config.model == 'roberta':
            self.encoder = RobertaModel.from_pretrained(config.roberta_path).to(config.device)
            self.encoder.resize_token_embeddings(config.vocab_size)
        if config.tune == 'prompt':
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.bert_word_embedding = self.encoder.get_input_embeddings()
        self.embedding_dim = self.bert_word_embedding.embedding_dim
        self.prompt_lens = config.prompt_len * config.prompt_num
        self.softprompt_encoder = nn.Embedding(self.prompt_lens, self.embedding_dim).to(config.device)
        # initialize prompt embedding
        self._init_prompt()
        self.prompt_ids = torch.LongTensor(list(range(self.prompt_lens))).to(self.config.device)

        self.info_nce_fc = nn.Linear(self.embedding_dim , self.embedding_dim).to(config.device)

    def _init_prompt(self):
        # is is is [e1] is is is [MASK] is is is [e2] is is is
        if self.config.prompt_init == 1:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            token_embedding = self.bert_word_embedding.weight[2003]
            prompt_embedding[list(range(self.prompt_lens)), :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data
       
        # ! @ # [e1] he is as [MASK] * & % [e2] just do it  
        elif self.config.prompt_init == 2:
            prompt_embedding = torch.zeros_like(self.softprompt_encoder.weight).to(self.config.device)
            ids = [999, 1030, 1001, 2002, 2003, 2004, 1008, 1004, 1003, 2074, 2079,  2009]
            for i in range(self.prompt_lens):
                token_embedding = self.bert_word_embedding.weight[ids[i]]
                prompt_embedding[i, :] = token_embedding.clone().detach()
            for param in self.softprompt_encoder.parameters():
                param.data = prompt_embedding # param.data


    def embedding_input(self, input_ids): # (b, max_len)
        input_embedding = self.bert_word_embedding(input_ids) # (b, max_len, h)
        prompt_embedding = self.softprompt_encoder(self.prompt_ids) # (prompt_len, h)

        for i in range(input_ids.size()[0]):
            p = 0
            for j in range(input_ids.size()[1]):
                if input_ids[i][j] == self.config.prompt_token_ids:
                    input_embedding[i][j] = prompt_embedding[p]
                    p += 1

        return input_embedding

    def forward_mixup(self, inputs):
        inputs['ids'] = inputs['ids'].to(self.config.device)
        inputs['mask'] = inputs['mask'].to(self.config.device)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size)

        # Get the encoder outputs
        outputs_words = self.encoder(inputs['ids'], attention_mask=inputs['mask'])[0]  # (batch_size, seq_len, hidden_size)
        
        # Find the positions of the [MASK] tokens (token ID is usually 103 for BERT models)
        mask_token_id = 103  # Assuming BERT-like tokenizer where [MASK] is 103
        mask_positions = (inputs['ids'] == mask_token_id)  # (batch_size, seq_len), True where [MASK] tokens are
        # Now, get the hidden states for the [MASK] tokens
        mask_hidden_states = outputs_words[mask_positions]  # (batch_size * 2, hidden_size)
        # Since each sample has exactly 2 [MASK] tokens, reshape it accordingly
        mask_hidden_states = mask_hidden_states.view(batch_size, 2, -1)  # (batch_size, 2, hidden_size)
        # Split the 2 [MASK] hidden states into masks_1 and masks_2
        masks_1 = mask_hidden_states[:, 0, :]  # (batch_size, hidden_size)
        masks_2 = mask_hidden_states[:, 1, :]  # (batch_size, hidden_size)
        return masks_1, masks_2
    
    def forward(self, inputs, is_des=False, is_distill=False, top_k=10): # (b, max_length)
        batch_size = inputs['ids'].size()[0]
        tensor_range = torch.arange(batch_size) # (b)     
        pattern = self.config.pattern
        if pattern == 'softprompt' or pattern == 'hybridprompt':
            input_embedding = self.embedding_input(inputs['ids'])
            outputs = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'], output_attentions=True)
            outputs_words = outputs[0]  # (b, max_length, h)
            attention_scores = outputs[2]
        else:
            outputs = self.encoder(inputs_embeds=input_embedding, attention_mask=inputs['mask'], output_attentions=True)
            outputs_words = outputs[0]  # (b, max_length, h)
            attention_scores = outputs[2]
            # outputs_words_des = self.encoder(inputs['ids_des'], attention_mask=inputs['mask_des'])[0] # (b, max_length, h)

        if is_distill:
            # attentions_layer: (B, H, S, S)
            attentions_layer = attention_scores[-1]  # choose last layer

            # Column-wise sum over queries (dim=2). Result: (B, H, S)
            col_sum_per_head = attentions_layer.sum(dim=2)

            # Aggregate across heads -> (B, S)
            token_scores = col_sum_per_head.mean(dim=1)  # (B, S)

            # Mask out padding tokens (if you have attention_mask of shape (B, S))
            mask = inputs['mask'].to(token_scores.dtype)            # convert to float
            token_scores = token_scores * mask                      # zero-out pad positions

            # Normalize to probabilities per example
            token_probs = token_scores / (token_scores.sum(dim=1, keepdim=True) + 1e-12)

            # Choose safe k (don't request more than seq length)
            S = token_probs.size(1)
            k = top_k if S > top_k else S

            # top-k indices and scores per example
            topk_scores, topk_indices = torch.topk(token_probs, k=k, dim=1)

            B, _, H = outputs_words.size()

            # 8) gather hidden vectors for top-k indices -> (B, k, H)
            topk_hidden_indices = topk_indices.unsqueeze(-1).expand(-1, -1, H)   # (B, k, H)
            # topk_hidden = torch.gather(outputs_words, dim=1, index=topk_hidden_indices)  # (B, k, H)


        # return [CLS] hidden
        if pattern == 'cls' or pattern == 'softprompt':
            clss = torch.zeros(batch_size, dtype=torch.long)
            return outputs_words[tensor_range ,clss] # (b, h)

        # return [MASK] hidden
        elif pattern == 'hardprompt' or pattern == 'hybridprompt':
            masks = []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                try:
                    mask = np.argwhere(ids == self.config.mask_token_ids)[0][0]
                except:
                    mask = 0
                
                masks.append(mask)
            if is_des:
                average_outputs_words = torch.mean(outputs_words, dim=1)
                return average_outputs_words
            elif is_distill:
                mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
                return mask_hidden, outputs_words, topk_hidden_indices
            else:
                mask_hidden = outputs_words[tensor_range, torch.tensor(masks)] # (b, h)
                return mask_hidden
    
        # return e1:e2 hidden
        elif pattern == 'marker':
            h1, t1 = [], []
            for i in range(batch_size):
                ids = inputs['ids'][i].cpu().numpy()
                h1_index, t1_index = np.argwhere(ids == self.config.h_ids), np.argwhere(ids == self.config.t_ids)
                h1.append(0) if h1_index.size == 0 else h1.append(h1_index[0][0])
                t1.append(0) if t1_index.size == 0 else t1.append(t1_index[0][0])

            h_state = outputs_words[tensor_range, torch.tensor(h1)] # (b, h)
            t_state = outputs_words[tensor_range, torch.tensor(t1)]

            concerate_h_t = (h_state + t_state) / 2 # (b, h)
            return concerate_h_t

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
        self.old_model = EncodingModel(self.config)
        self.old_model.load_state_dict(self.history["state_dict"])
        self.old_model.eval()
        self.old_model.to(self.config.device)
        return self.old_model
    
    def cleanup_old_model(self):
        """Clean up the old model to free memory"""
        if hasattr(self, 'old_model') and self.old_model is not None:
            del self.old_model
            self.old_model = None
            torch.cuda.empty_cache() if torch.cuda.is_available() else None