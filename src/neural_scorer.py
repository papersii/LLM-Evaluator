import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
import os

class EvaluationDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Format: [CLS] Question [SEP] Reference [SEP] Candidate [SEP]
        text_a = f"{item['question']} [SEP] {item['answer']}"
        text_b = item['response']
        
        encoding = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        label = torch.tensor(item['label'], dtype=torch.long)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': label
        }

class NeuralScorer:
    def __init__(self, model_name='bert-base-uncased', device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
        self.model.to(self.device)

    def train(self, train_data, output_dir='neural_scorer_model', epochs=3, batch_size=8, lr=2e-5):
        dataset = EvaluationDataset(train_data, self.tokenizer)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)

        self.model.train()
        os.makedirs(output_dir, exist_ok=True)

        print(f"Starting training on {self.device} for {epochs} epochs...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

        # Save model
        print(f"Saving model to {output_dir}")
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def predict(self, question, reference, candidate):
        self.model.eval()
        text_a = f"{question} [SEP] {reference}"
        text_b = candidate
        
        inputs = self.tokenizer(
            text_a, 
            text_b, 
            truncation=True, 
            padding=True, 
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            
        return bool(prediction == 1), probs[0][1].item() # Returns (is_correct, confident_score)
