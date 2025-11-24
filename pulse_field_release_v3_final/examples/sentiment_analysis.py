
import torch
import torch.nn as nn
import torch.optim as optim
from core.impulse import ImpulseEncoder, ImpulseDecoder, Impulse
from core.crystals import Crystal

def real_training_demo():
    print("--- Starting Real Training Demo ---")
    
    # Hyperparameters
    vocab_size = 1000
    dim = 128
    lr = 0.01
    epochs = 100
    
    # Initialize Real Components
    encoder = ImpulseEncoder(vocab_size=vocab_size, dim=dim)
    crystal = Crystal(crystal_id="processor_1", input_dim=dim, output_dim=dim)
    decoder = ImpulseDecoder(dim=dim, vocab_size=2) # Binary classification
    
    # Optimizer
    params = list(encoder.parameters()) + list(crystal.parameters()) + list(decoder.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Dataset: Simple Sentiment Analysis (Mock)
    # "good" -> Class 1, "bad" -> Class 0
    data = [
        ("this is good", 1),
        ("very good job", 1),
        ("good results", 1),
        ("this is bad", 0),
        ("very bad error", 0),
        ("bad failure", 0),
    ]
    
    print(f"Training on {len(data)} samples for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        
        for text, label in data:
            optimizer.zero_grad()
            
            # 1. Encode (Real Embedding)
            impulse = encoder(text)
            
            # 2. Process (Real Linear Layer)
            processed_impulse, _ = crystal.process(impulse)
            
            # 3. Decode (Real Linear Head)
            logits = decoder(processed_impulse)
            
            # 4. Loss & Backprop
            target = torch.tensor([label], dtype=torch.long)
            loss = criterion(logits.unsqueeze(0), target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy check
            pred = torch.argmax(logits).item()
            if pred == label:
                correct += 1
        
        if epoch % 10 == 0:
            acc = correct / len(data)
            print(f"Epoch {epoch}: Loss={total_loss:.4f}, Accuracy={acc:.2f}")
            
    print("--- Training Complete ---")
    
    # Test
    test_text = "good job"
    impulse = encoder(test_text)
    processed, _ = crystal.process(impulse)
    logits = decoder(processed)
    pred = torch.argmax(logits).item()
    print(f"Test '{test_text}': Predicted Class {pred} (Expected 1)")
    
    test_text = "bad error"
    impulse = encoder(test_text)
    processed, _ = crystal.process(impulse)
    logits = decoder(processed)
    pred = torch.argmax(logits).item()
    print(f"Test '{test_text}': Predicted Class {pred} (Expected 0)")

if __name__ == "__main__":
    real_training_demo()
