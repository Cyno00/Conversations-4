import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import random
from collections import deque

GAME_MAX_SUBJECTS = 20
GAME_MAX_MEMORY_BANK = 10
NETWORK_CONV_HISTORY_ITEMS = 5

@dataclass
class TrainingExample:
    """Single training example from processed data"""
    game_id: str
    turn_number: int
    state_tensor: torch.Tensor
    action_index: int
    reward: float
    next_state_tensor: Optional[torch.Tensor]
    done: bool
	
class StateEncoder:
    """Converts JSON state to fixed-size tensor representation"""

    def __init__(self, max_subjects=GAME_MAX_SUBJECTS, max_memory_items=GAME_MAX_MEMORY_BANK, max_conversation_items=NETWORK_CONV_HISTORY_ITEMS):
        self.max_subjects = max_subjects
        self.max_memory_items = max_memory_items
        self.max_conversation_items = max_conversation_items
        
    def encode_state(self, state_json: Dict) -> torch.Tensor:
        """Convert state JSON to tensor"""
        features = []
        
        # 1. Encode conversation history
        conv_features = torch.zeros(self.max_conversation_items, self.max_subjects + 1)  # +1 for importance
        conversation = state_json['conversation_history']
        for i, item in enumerate(conversation[-self.max_conversation_items:]):
            # One-hot encode subjects
            for subject in item['subjects']:
                if subject < self.max_subjects:
                    conv_features[i, subject] = 1.0
            # Add importance
            conv_features[i, -1] = item['importance']
        
        # Flatten conversation features
        features.append(conv_features.flatten())
        
        # 2. Encode memory bank
        memory_features = torch.zeros(self.max_memory_items, self.max_subjects + 1)
        memory_bank = state_json['my_memory_bank']
        for i, item in enumerate(memory_bank[:self.max_memory_items]):
            # One-hot encode subjects
            for subject in item['subjects']:
                if subject < self.max_subjects:
                    memory_features[i, subject] = 1.0
            # Add importance
            memory_features[i, -1] = item['importance']
        
        # Flatten memory features
        features.append(memory_features.flatten())
        
        # 3. Encode preferences (ranking of subjects)
        pref_features = torch.zeros(self.max_subjects)
        preferences = state_json['my_preferences']
        for i, subject in enumerate(preferences):
            if subject < self.max_subjects:
                # Higher rank = higher value (inverse of position)
                pref_features[subject] = 1.0 - (i / len(preferences))
        features.append(pref_features)
        
        # 4. Encode game metadata
        metadata = torch.tensor([
            state_json['turn_number'] / state_json['conversation_length'],  # Progress
            len(state_json['conversation_history']) / self.max_conversation_items,  # Conv fullness
            state_json['pause_count'] / 3.0,  # Normalized pauses
            1.0 if state_json['game_over'] else 0.0  # Game over flag
        ])
        features.append(metadata)
        
        # Concatenate all features
        return torch.cat(features)
    
    def get_feature_size(self) -> int:
        """Calculate total feature vector size"""
        conv_size = self.max_conversation_items * (self.max_subjects + 1)
        memory_size = self.max_memory_items * (self.max_subjects + 1)
        pref_size = self.max_subjects
        metadata_size = 4
        return conv_size + memory_size + pref_size + metadata_size

class QNetwork(nn.Module):
    """Q-Network for action value estimation"""
    
    def __init__(self, state_size: int, max_actions: int, hidden_size: int = 512):
        super().__init__()
        self.max_actions = max_actions
        
        self.network = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, max_actions)
        )
        
    def forward(self, state: torch.Tensor, num_actions: int) -> torch.Tensor:
        """
        Forward pass - returns Q-values for available actions
        Args:
            state: State tensor
            num_actions: Number of available actions (memory_bank_size + 1)
        """
        q_values = self.network(state)
        # Only return Q-values for available actions
        return q_values[:num_actions]

class DataLoader:
    """Loads and processes training data from JSON files"""
    
    def __init__(self, data_directory: str):
        self.data_directory = Path(data_directory)
        self.state_encoder = StateEncoder()
        
    def load_all_data(self) -> List[TrainingExample]:
        """Load all training examples from batch files"""
        training_examples = []
        
        # Find all batch files
        batch_files = sorted(self.data_directory.glob("batch_*.json"))
        print(f"Found {len(batch_files)} batch files")
        
        for batch_file in batch_files:
            print(f"Loading {batch_file.name}...")
            try:
                with open(batch_file, 'r') as f:
                    games_data = json.load(f)
                
                # Process each game
                for game_data in games_data:
                    game_examples = self._process_game(game_data)
                    training_examples.extend(game_examples)
                    
            except Exception as e:
                print(f"Error loading {batch_file}: {e}")
        
        print(f"Loaded {len(training_examples)} training examples")
        return training_examples
    
    def _process_game(self, game_data: Dict) -> List[TrainingExample]:
        """Process a single game's training examples"""
        examples = []
        training_data = game_data['training_examples']
        
        for i, example in enumerate(training_data):
            # Encode current state
            state_tensor = self.state_encoder.encode_state(example['state_before'])
            
            # Convert action to index
            action_index = self._action_to_index(example['our_action'], example['state_before'])
            
            # Get next state (if available)
            next_state_tensor = None
            if i + 1 < len(training_data):
                next_example = training_data[i + 1]
                # Only use as next state if consecutive turns
                if next_example['turn_number'] == example['turn_number'] + 1:
                    next_state_tensor = self.state_encoder.encode_state(next_example['state_before'])
            
            training_example = TrainingExample(
                game_id=example.get('game_id', 'unknown'),
                turn_number=example['turn_number'],
                state_tensor=state_tensor,
                action_index=action_index,
                reward=example['immediate_reward'],
                next_state_tensor=next_state_tensor,
                done=example['game_over']
            )
            
            examples.append(training_example)
        
        return examples
    
    def _action_to_index(self, action: Dict, state: Dict) -> int:
        """Convert action to index in memory bank"""
        if action['action_type'] == 'pass':
            # Pass action is index = len(memory_bank)
            return len(state['my_memory_bank'])
        else:
            # Find which memory bank item was proposed
            item_id = action['item_id']
            for i, memory_item in enumerate(state['my_memory_bank']):
                if memory_item['item_id'] == item_id:
                    return i
            # Fallback to pass if item not found
            return len(state['my_memory_bank'])

class QLearningTrainer:
    """Q-Learning trainer"""
    
    def __init__(self, state_size: int, max_actions: int, lr: float = 1e-4, gamma: float = 0.95):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.q_network = QNetwork(state_size, max_actions).to(self.device)
        self.target_network = QNetwork(state_size, max_actions).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        self.gamma = gamma
        self.update_target_every = 1000  # Update target network every N steps
        self.training_step = 0
        
        # Copy initial weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def train(self, training_examples: List[TrainingExample], epochs: int = 50, batch_size: int = 32):
        """Train the Q-network"""
        print(f"Training for {epochs} epochs with batch size {batch_size}")
        
        # Filter examples that have next states (for proper Q-learning)
        valid_examples = [ex for ex in training_examples if ex.next_state_tensor is not None]
        print(f"Using {len(valid_examples)} examples with next states for Q-learning")
        
        for epoch in range(epochs):
            # Shuffle training data
            random.shuffle(valid_examples)
            
            total_loss = 0.0
            num_batches = 0
            
            # Process in batches
            for i in range(0, len(valid_examples), batch_size):
                batch = valid_examples[i:i + batch_size]
                loss = self._train_batch(batch)
                total_loss += loss
                num_batches += 1
                
                # Update target network
                if self.training_step % self.update_target_every == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())
                
                self.training_step += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    def _train_batch(self, batch: List[TrainingExample]) -> float:
        """Train on a single batch"""
        if not batch:
            return 0.0
        
        # Prepare batch data
        states = torch.stack([ex.state_tensor for ex in batch]).to(self.device)
        actions = torch.tensor([ex.action_index for ex in batch]).to(self.device)
        rewards = torch.tensor([ex.reward for ex in batch], dtype=torch.float).to(self.device)
        next_states = torch.stack([ex.next_state_tensor for ex in batch]).to(self.device)
        dones = torch.tensor([ex.done for ex in batch], dtype=torch.bool).to(self.device)
        
        # Current Q-values
        current_q_values = []
        for i, ex in enumerate(batch):
            memory_size = len(ex.state_tensor)  # Simplified - should extract actual memory size
            num_actions = GAME_MAX_MEMORY_BANK + 1
            q_vals = self.q_network(states[i:i+1], num_actions)
            # print(f"Q-values for example {i}: {q_vals}")
            # print(f"Action taken: {actions[i]}, Q-value: {q_vals[0, actions[i]].item()}")
            current_q_values.append(q_vals[0, actions[i]])
        
        current_q_values = torch.stack(current_q_values)
        
        # Target Q-values using target network
        with torch.no_grad():
            target_q_values = []
            for i, ex in enumerate(batch):
                if dones[i]:
                    target_q_values.append(rewards[i])
                else:
                    num_actions = GAME_MAX_MEMORY_BANK + 1
                    next_q_vals = self.target_network(next_states[i:i+1], num_actions)
                    target = rewards[i] + self.gamma * next_q_vals.max()
                    target_q_values.append(target)
            
            target_q_values = torch.stack(target_q_values)
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str):
        """Save trained model"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        print(f"Model loaded from {path}")

def main():
    """Main training function"""
    if len(sys.argv) != 2:
        print("Usage: uv run players/player_4/train.py <data_directory>")
        print("Example: uv run players/player_4/train.py players/player_4/processed/")
        return
    
    data_directory = sys.argv[1]
    
    # Check if directory exists
    if not Path(data_directory).exists():
        print(f"Error: Directory {data_directory} not found!")
        return
    
    print("=" * 60)
    print("Q-LEARNING TRAINING FOR CONVERSATION GAME")
    print("=" * 60)
    
    # Step 1: Load and process data
    print("\n1. LOADING DATA...")
    data_loader = DataLoader(data_directory)
    training_examples = data_loader.load_all_data()
    
    if not training_examples:
        print("No training examples found!")
        return
    
    # Step 2: Initialize trainer
    print("\n2. INITIALIZING Q-NETWORK...")
    state_size = data_loader.state_encoder.get_feature_size()
    max_actions = GAME_MAX_MEMORY_BANK + 1  # + 1 for pass action
    
    print(f"State vector size: {state_size}")
    print(f"Maximum actions: {max_actions}")
    
    trainer = QLearningTrainer(state_size, max_actions, lr=1e-5, gamma=0.95)
    
    # Step 3: Train the model
    print("\n3. TRAINING Q-NETWORK...")
    trainer.train(training_examples, epochs=100, batch_size=32)
    
    # Step 4: Save the model
    print("\n4. SAVING MODEL...")
    model_path = Path(data_directory).parent / "trained_qnetwork.pth"
    trainer.save_model(str(model_path))
    
    print(f"\n✓ Training complete! Model saved to: {model_path}")
    print("\nYou can now use this trained model to play games!")

if __name__ == "__main__":
    main()