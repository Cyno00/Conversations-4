#!/usr/bin/env python3
"""
process.py - Convert raw game JSON files to processed training data

Usage: python process.py
- Reads from raw/batch_*.json  
- Automatically identifies Player4 by looking at speaker_name
- Extracts training examples ONLY when Player4 acts (proposes item or passes)
- Game state includes ONLY Player4's memory bank and preferences (realistic)
- Saves to processed/batch_*.json
- Skips games where Player4 never speaks
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ProcessedItem:
    """Processed item representation"""
    subjects: List[int]
    importance: float
    item_id: str

@dataclass
class ProcessedGameState:
    """Game state at a specific turn - Player4's view only"""
    turn_number: int
    conversation_history: List[ProcessedItem]
    my_memory_bank: List[ProcessedItem]  # Only Player4's memory bank
    my_preferences: List[int]  # Only Player4's preferences
    conversation_length: int
    pause_count: int
    game_over: bool

@dataclass
class ProcessedAction:
    """Action taken by a player"""
    player_id: str
    action_type: str  # "propose_item", "pass"
    item_id: Optional[str]  # None if pass
    item_subjects: Optional[List[int]]  # None if pass
    item_importance: Optional[float]  # None if pass

@dataclass
class TrainingExample:
    """Training example for Player4"""
    game_id: str
    turn_number: int
    state_before: ProcessedGameState
    our_action: ProcessedAction
    immediate_reward: float
    game_over: bool

@dataclass
class ProcessedGame:
    """Complete processed game - focused on Player4 training data"""
    game_id: str
    player4_id: str
    training_examples: List[TrainingExample]  # Training data for Player4
    final_scores: Dict[str, Dict[str, float]]
    game_metadata: Dict[str, Any]

class GameDataProcessor:
    """Processes raw game JSON into structured format"""
    
    def __init__(self):
        self.processed_games = []
    
    def find_player4_id(self, raw_game: Dict) -> Optional[str]:
        """
        First pass: Find Player4's ID by looking at speaker_name
        Returns None if Player4 never speaks
        """
        for turn_data in raw_game['turn_impact']:
            speaker_name = turn_data.get('speaker_name', '')
            if speaker_name == 'Player4':
                return turn_data['speaker_id']
        return None
    
    def process_raw_game(self, raw_game: Dict) -> Optional[ProcessedGame]:
        """Convert raw game JSON to ProcessedGame"""
        
        # First pass: Find Player4's ID
        player4_id = self.find_player4_id(raw_game)
        if player4_id is None:
            print("  Skipping game - Player4 never speaks")
            return None
        
        print(f"  Found Player4 ID: {player4_id[:8]}...")
        
        # Extract player information
        players = {p['id']: p for p in raw_game['scores']['player_scores']}
        
        # Check if Player4 is in the final scores
        if player4_id not in players:
            print(f"  Skipping game - Player4 {player4_id} not found in player scores")
            return None
        
        # Second pass: Process each turn, but only extract training data when Player4 acts
        training_examples = []
        conversation_history = []
        
        for turn_data in raw_game['turn_impact']:
            # Player4 makes a decision every turn: either propose an item or pass
            # We create training examples for every turn where Player4 had to decide
            player4_action = None
            
            if player4_id in turn_data['proposals']:
                # Player4 chose to propose an item
                proposal = turn_data['proposals'][player4_id]
                player4_action = ProcessedAction(
                    player_id=player4_id,
                    action_type="propose_item",
                    item_id=proposal['id'],
                    item_subjects=proposal['subjects'],
                    item_importance=proposal['importance']
                )
            else:
                # Player4 chose to pass 
                # This happens whether others proposed or no one proposed
                player4_action = ProcessedAction(
                    player_id=player4_id,
                    action_type="pass",
                    item_id=None,
                    item_subjects=None,
                    item_importance=None
                )
            
            # Create game state before this turn - Player4's view only
            game_state_before = ProcessedGameState(
                turn_number=turn_data['turn'],
                conversation_history=[
                    ProcessedItem(
                        subjects=item['subjects'],
                        importance=item['importance'],
                        item_id=item['id']
                    ) for item in conversation_history
                ],
                my_memory_bank=[
                    ProcessedItem(
                        subjects=item['subjects'],
                        importance=item['importance'],
                        item_id=item['id']
                    ) for item in players[player4_id]['memory_bank']
                ],
                my_preferences=players[player4_id]['preferences'],
                conversation_length=raw_game['scores']['conversation_length'],
                pause_count=raw_game['scores']['pauses'],
                game_over=turn_data['is_over']
            )
            
            # Update conversation history for next turn
            if turn_data.get('item'):
                conversation_history.append(turn_data['item'])
            
            # Calculate immediate reward for Player4's decision
            immediate_reward = 0.0
            if turn_data['speaker_id'] == player4_id:
                immediate_reward = turn_data['score_impact']['total']
            elif player4_action.action_type == "propose_item":
                immediate_reward = self._calculate_individual_bonus(
                        turn_data['item']['subjects'],
                        players[player4_id]['preferences']
                    )
            else:
                other_player_spoken_item_subjects = turn_data.get('item', {}).get('subjects', [])
                if other_player_spoken_item_subjects:
                    immediate_reward = self._calculate_individual_bonus(
                        other_player_spoken_item_subjects,
                        players[player4_id]['preferences']
                    )
                else:
                    immediate_reward = 0.0
                    
            training_example = TrainingExample(
                game_id=f"game_{len(self.processed_games)}",
                turn_number=turn_data['turn'],
                state_before=game_state_before,
                our_action=player4_action,
                immediate_reward=immediate_reward,
                game_over=turn_data['is_over']
            )
            
            training_examples.append(training_example)
            
        
        # Create final processed game
        processed_game = ProcessedGame(
            game_id=f"game_{len(self.processed_games)}",
            player4_id=player4_id,
            training_examples=training_examples,
            final_scores={
                pid: player_data['scores']
                for pid, player_data in players.items()
            },
            game_metadata={
                'player4_id': player4_id,
                'player4_training_examples': len(training_examples),
                'conversation_length': raw_game['scores']['conversation_length'],
                'total_pauses': raw_game['scores']['pauses'],
                'num_players': len(players),
                'shared_score_breakdown': raw_game['score_breakdown']
            }
        )
        
        return processed_game
    
    def _calculate_individual_bonus(self, subjects: List[int], preferences: List[int]) -> float:
        """Calculate individual preference bonus for subjects"""
        if not subjects:
            return 0.0
        
        bonus = 0.0
        for subject in subjects:
            if subject in preferences:
                rank = preferences.index(subject)
                bonus += 1.0 - (rank / len(preferences))
        return bonus / len(subjects)

def process_batch_file(input_path: Path, output_path: Path) -> bool:
    """Process a single batch file"""
    try:
        print(f"Processing {input_path.name}...")
        
        # Load raw data
        with open(input_path, 'r') as f:
            raw_data = json.load(f)
        
        # Initialize processor
        processor = GameDataProcessor()
        
        # Process each game in the batch
        processed_games = []
        skipped_games = 0
        
        if isinstance(raw_data, list):
            # Multiple games in one file
            for i, raw_game in enumerate(raw_data):
                try:
                    processed_game = processor.process_raw_game(raw_game)
                    if processed_game is not None:
                        processed_game.game_id = f"{input_path.stem}_game_{i}"
                        processed_games.append(asdict(processed_game))
                    else:
                        skipped_games += 1
                except Exception as e:
                    print(f"  Error processing game {i} in {input_path.name}: {e}")
                    skipped_games += 1
        else:
            # Single game in file
            try:
                processed_game = processor.process_raw_game(raw_data)
                if processed_game is not None:
                    processed_game.game_id = input_path.stem
                    processed_games.append(asdict(processed_game))
                else:
                    skipped_games += 1
            except Exception as e:
                print(f"  Error processing {input_path.name}: {e}")
                return False
        
        if not processed_games:
            print(f"  No valid games found in {input_path.name} (Player4 never spoke in any games)")
            return False
        
        # Save processed data
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(processed_games, f, indent=2)
        
        total_training_examples = sum(len(game['training_examples']) for game in processed_games)
        print(f"  Success: {len(processed_games)} games, {total_training_examples} training examples")
        if skipped_games > 0:
            print(f"  Skipped: {skipped_games} games (Player4 didn't speak)")
        
        return True
        
    except Exception as e:
        print(f"  Failed to process {input_path.name}: {e}")
        return False

def main():
    """Main processing function"""
    raw_dir = Path("players/player_4/raw")
    processed_dir = Path("players/player_4/processed")
    
    # Check if raw directory exists
    if not raw_dir.exists():
        print(f"Error: {raw_dir} directory not found!")
        print("Please create the 'raw' directory and place your batch_*.json files there.")
        return
    
    # Find all batch files
    batch_files = sorted(raw_dir.glob("batch_*.json"))
    
    if not batch_files:
        print(f"No batch_*.json files found in {raw_dir}/")
        print("Expected files like: batch_1.json, batch_2.json, etc.")
        return
    
    print(f"Found {len(batch_files)} batch files to process")
    print("Looking for Player4 in each game...")
    
    # Process each batch file
    success_count = 0
    for batch_file in batch_files:
        output_file = processed_dir / batch_file.name
        
        if process_batch_file(batch_file, output_file):
            success_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count}/{len(batch_files)} files")
    print(f"Processed files saved to: {processed_dir}/")
    
    # Show summary
    if success_count > 0:
        print(f"\nProcessed file structure for Player4:")
        total_training_examples = 0
        total_games = 0
        for processed_file in sorted(processed_dir.glob("batch_*.json")):
            try:
                with open(processed_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        game_count = len(data)
                        training_count = sum(len(game.get('training_examples', [])) for game in data)
                    else:
                        game_count = 1
                        training_count = len(data.get('training_examples', []))
                    total_training_examples += training_count
                    total_games += game_count
                    print(f"  {processed_file.name}: {game_count} games, {training_count} training examples")
            except:
                print(f"  {processed_file.name}: (error reading)")
        
        print(f"\nTotal: {total_games} games processed, {total_training_examples} training examples extracted")
        print(f"Ready for RL training!")
        
        if total_training_examples == 0:
            print(f"\nWARNING: No training examples found for Player4")
            print("Player4 may not be speaking in any games, or there may be an issue with the data format.")

if __name__ == "__main__":
    main()