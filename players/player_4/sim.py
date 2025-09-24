import asyncio
import json
import yaml
from datetime import datetime
from pathlib import Path
import multiprocessing
import time
import random

async def run_simulation(scenario_name, args, run_index, semaphore):
    """Run a single simulation"""
    cmd = ['uv', 'run', 'main.py']
    
    # Generate random seed for this run
    run_seed = random.randint(0, 2**31 - 1)
    cmd.extend(['--seed', str(run_seed)])
    
    # Add simple arguments
    for key in ['subjects', 'memory_size', 'length']:
        if key in args:
            cmd.extend([f'--{key}', str(args[key])])
    
    # Add player arguments
    if 'players' in args:
        for player_type, count in args['players']:
            cmd.extend(['--player', player_type, str(count)])
    
    async with semaphore:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, _ = await proc.communicate()
        
        # Parse JSON output
        json_output = "{" + stdout.decode().split("{", 1)[-1]
        result = json.loads(json_output)
        
         # Filter scores to only keep player_scores with id and scores fields
        filtered_scores = {}
        if 'scores' in result:
            if 'player_scores' in result['scores']:
                filtered_scores['player_scores'] = [
					{'id': player['id'], 'scores': player['scores']} 
					for player in result['scores']['player_scores']
				]
            if "shared_score_breakdown" in result['scores']:
                filtered_scores['shared_score_breakdown'] = result['scores']['shared_score_breakdown']
        
        # Keep only turn_impact and filtered scores
        filtered_result = {
            # 'turn_impact': result.get('turn_impact'),
            'scores': filtered_scores
        }
        
        return {
            'scenario': scenario_name,
            'run': run_index,
            'seed': run_seed,
            'args': args,
            'result': filtered_result
        }
        
async def run_scenario(scenario, semaphore):
    """Run all iterations of a single scenario"""
    name = scenario['name']
    runs = scenario['runs']
    
    print(f"Starting scenario '{name}' ({runs} runs)...")
    start = time.time()
    
    # Run all iterations of this scenario
    results = await asyncio.gather(*[
        run_simulation(name, scenario['args'], i, semaphore)
        for i in range(runs)
    ])
    
    elapsed = time.time() - start
    print(f"  Completed '{name}': {runs} runs in {elapsed:.1f}s ({elapsed/runs}s per run)")
    
    return results

async def main():
    import sys
    
    # Load config
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'players/player_4/scenarios.yaml'
    with open(config_file) as f:
        scenarios = yaml.safe_load(f)['scenarios']
    
    # Create all tasks
    tasks = []
    for scenario in scenarios:
        for i in range(scenario['runs']):
            tasks.append((scenario['name'], scenario['args'], i))
    
    print(f"Running {len(tasks)} simulations on {multiprocessing.cpu_count()*2} parallel processes...")
    start_time = time.time()
    
    # Run with concurrency limit
    semaphore = asyncio.Semaphore(multiprocessing.cpu_count() * 2)
    
    # Run each scenario and flatten results
    all_results = []
    for scenario in scenarios:
        scenario_results = await run_scenario(scenario, semaphore)
        all_results.extend(scenario_results)
    
    elapsed = time.time() - start_time
    print(f"\nAll scenarios complete!")
    
    # Save results
    output_dir = Path('players/player_4/results')
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f'results_{timestamp}.json'
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Saved {len(all_results)} results to {output_file}")

if __name__ == '__main__':
    asyncio.run(main())