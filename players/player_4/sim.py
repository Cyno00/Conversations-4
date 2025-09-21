import asyncio
import json
import aiofiles
from collections import deque
import os

class StreamingSimulator:
    def __init__(self, total_games=512, buffer_size=128, concurrent_games=32):
        self.buffer = deque(maxlen=buffer_size)
        self.total_games = total_games
        self.total_games_so_far = 0
        self.concurrent_games = concurrent_games

    async def run_continuous(self):
        """Continuously generate games and save batches"""
        while True:
            # Run game asynchronously
            tasks = [self.run_game_async() for _ in range(self.concurrent_games)]
            results = await asyncio.gather(*tasks)
            for game_data in results:
                self.buffer.append(game_data)
                self.total_games_so_far += 1
            print(f'Total games simulated: {self.total_games_so_far}')

            # Save batch periodically
            if len(self.buffer) >= self.buffer.maxlen:
                await self.save_batch()
                print(f'Saved batch of {self.buffer.maxlen} games.')
                self.buffer.clear()
            
            # Stop if we've reached the total number of games
            if self.total_games_so_far >= self.total_games:
                break
    
    async def run_game_async(self):
        """Run game simulator asynchronously"""
        proc = await asyncio.create_subprocess_exec(
            'uv',
            'run',
            'main.py',
            '--subjects', '20',
            '--memory_size', '10',
            '--length', '100',
            '--player', 'prp', '9',
            '--player', 'p4', '1',
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await proc.communicate()
        # Remove everything from stdout before the first { character
        json_output = "{" + stdout.decode().split("{", 1)[-1]
        return json.loads(json_output)

    async def save_batch(self):
        """Save current buffer to file"""
        filename = f'players/player_4/raw/batch_{self.total_games_so_far // self.buffer.maxlen}.json'
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        async with aiofiles.open(filename, 'w') as f:
            await f.write(json.dumps(list(self.buffer)))
            print(f'Saved batch to {filename}')

streaming_simulator = StreamingSimulator()
asyncio.run(streaming_simulator.run_continuous())

