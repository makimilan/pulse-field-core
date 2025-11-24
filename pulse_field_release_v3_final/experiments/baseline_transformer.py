import torch
import torch.nn as nn
import time
import numpy as np
import json

class SimpleTransformer(nn.Module):
    """
    A simple GPT-style Transformer baseline for benchmarking.
    """
    def __init__(self, vocab_size=50257, d_model=768, n_head=12, n_layer=6, max_len=1024):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=d_model*4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.num_params = sum(p.numel() for p in self.parameters())

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        x = self.transformer_encoder(x)
        logits = self.lm_head(x)
        return logits

class BaselineExperiment:
    def run(self):
        print("Running Baseline Transformer Experiment (Real Training)...")
        # Use smaller model for quick demo training
        model = SimpleTransformer(vocab_size=1000, d_model=64, n_head=2, n_layer=2, max_len=128)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        steps = 100
        history = {
            "step": [],
            "ppl": [],
            "accuracy": [],
            "latency": []
        }
        
        # Generate dummy data
        data = torch.randint(0, 1000, (steps, 32)) # (steps, seq_len)
        
        model.train()
        for i in range(steps):
            inputs = data[i].unsqueeze(0) # (1, seq_len)
            targets = inputs.clone()
            
            start_time = time.time()
            optimizer.zero_grad()
            logits = model(inputs) # (1, seq_len, vocab_size)
            
            # Shift logits and targets for LM task
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()
            
            loss = criterion(shift_logits.view(-1, 1000), shift_labels.view(-1))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            
            latency = (end_time - start_time) * 1000 # ms
            ppl = torch.exp(loss).item()
            
            # Calculate accuracy
            preds = torch.argmax(shift_logits, dim=-1)
            acc = (preds == shift_labels).float().mean().item()
            
            if i % 10 == 0:
                history["step"].append(i)
                history["ppl"].append(ppl)
                history["accuracy"].append(acc)
                history["latency"].append(latency)
                print(f"Step {i}: Loss={loss.item():.4f}, PPL={ppl:.2f}, Acc={acc:.2f}, Latency={latency:.2f}ms")
            
        # Ensure reports directory exists
        import os
        os.makedirs("reports", exist_ok=True)
        with open("reports/baseline_results.json", "w") as f:
            json.dump(history, f, indent=2)
            
        return history

if __name__ == "__main__":
    exp = BaselineExperiment()
    exp.run()
