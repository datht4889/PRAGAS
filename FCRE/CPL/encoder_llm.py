import os
import torch
import torch.nn as nn
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional
from llm2vec import LLM2Vec


class EncodingModel_LLM2vec(nn.Module):
    def __init__(self, config):
        nn.Module.__init__(self)
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
                "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            )
        self.encoder = LLM2Vec.from_pretrained(
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp",
            peft_model_name_or_path="McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse",
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
            merge_peft=True,
            pooling_mode="mean",
            max_length=512,
            skip_instruction = False,
            token=os.getenv('HUGGINGFACE_API_KEY')
        )
        # merge LoRA of encoder


        self.encoder.model = self.initialize_peft(
            self.encoder.model,
        )
        # set dtype of vector_linear to bfloat16

    def initialize_peft(
        self,
        model,
        lora_r: int = 64,
        lora_alpha: int = 128,
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

    def forward(self, inputs, is_augment=False, is_distill=False, top_k=10): # (b, max_length)
        batch_size = len(inputs)
        input_data = self.encoder.tokenize(inputs)
        input_data = {k: v.cuda() for k, v in input_data.items()}

        # Get model outputs with attention scores
        outputs = self.encoder.model(
            input_ids=input_data['input_ids'],
            attention_mask=input_data['attention_mask'],
            output_attentions=True if is_distill else None
        )


        # Get last hidden states
        last_hidden_state = outputs.last_hidden_state
        attention_scores = outputs.attentions

        embeddings = self.encoder.get_pooling(input_data, last_hidden_state)  # Shape: [batch_size, hidden_size]
        assert embeddings.shape[0] == batch_size, f"{embeddings.shape[0]} != {batch_size}, {embeddings.shape}, {last_hidden_state.shape}"

        if is_distill:
            # Get the last layer attention scores (similar to EncodingModel)
            attentions_layer = attention_scores[-1]  # choose last layer

            # Column-wise sum over queries (dim=2). Result: (B, H, S)
            col_sum_per_head = attentions_layer.sum(dim=2)

            # Aggregate across heads -> (B, S)
            token_scores = col_sum_per_head.mean(dim=1)  # (B, S)

            # Mask out padding tokens
            attention_mask = input_data['attention_mask']
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

        # For augmentation mode, you could implement similar to forward_mixup
        if is_augment:
            # This part depends on how you want to handle augmentation
            # You might need to adapt your forward_mixup logic here
            raise NotImplementedError("Augmentation mode not implemented yet")

        return embeddings

    def forward_mixup(self, inputs):
        """
        inputs: batch of [input1, input2]
        """
        batch_size = len(inputs)
        merged_inputs = []
        for i in range(batch_size):
            merged_inputs.append(inputs[i][0] + " " + inputs[i][1])
        input_data = self.tokenizer(merged_inputs, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_data = {k: v.cuda() for k, v in input_data.items()}
        # take tokens from inputs[i][0] and inputs[i][1]
        outputs = self.encoder.model(**input_data)
        last_hidden_state = outputs.last_hidden_state
        temp = [inputs[i][0] for i in range(batch_size)]
        input_data_1 = self.tokenizer(temp)
        len_first = [len(input_data_1['input_ids'][i]) for i in range(batch_size)]

        last_token_indices = input_data['attention_mask'].sum(dim=1) - 1
        last_token_indices = last_token_indices.tolist()

        first = [last_hidden_state[i, :len_first[i], :].mean(dim = 0) for i in range(batch_size)]
        second = [last_hidden_state[i, len_first[i]:last_token_indices[i], :].mean(dim = 0) for i in range(batch_size)]
        first = torch.stack(first)
        second = torch.stack(second)
        # compress to bert size
        return first, second

    def compress_to_bert_size(self, vectors):
        return self.vector_linear(vectors)

    def save_lora_weights(self, save_path):
        """
        Save the LoRA weights of the model.

        Args:
            save_path (str): The path where the LoRA weights will be saved. (.../encoder)
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()

        # Get the underlying PEFT model
        peft_model = self.encoder.model

        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be saved.")

        # Save only the LoRA weights
        peft_model.save_pretrained(save_path)

        # Save the vector linear weights
        torch.save(self.vector_linear.state_dict(), os.path.join(save_path, "vector_linear.pth"))

        print(f"LoRA weights saved to {save_path}")

    def load_lora_weights(self, load_path):
        """
        Load the LoRA weights into the model.

        Args:
            load_path (str): The path where the LoRA weights are saved.
        """
        # Ensure the encoder model is in evaluation mode
        self.encoder.model.eval()

        # Get the underlying PEFT model
        peft_model = self.encoder.model

        # Check if the model is a PeftModel
        if not isinstance(peft_model, PeftModel):
            raise ValueError("The model doesn't seem to be a PeftModel. LoRA weights cannot be loaded.")

        # Load the LoRA weights
        peft_model.load_adapter(load_path, adapter_name="default")
        # Load the vector linear weights
        self.vector_linear.load_state_dict(torch.load(os.path.join(load_path, "vector_linear.pth")))
        print(f"LoRA weights loaded from {load_path}")

        # Optionally, you can set the loaded adapter as active
        peft_model.set_adapter("default")


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
        self.old_model = EncodingModel_LLM2vec(self.config)
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
