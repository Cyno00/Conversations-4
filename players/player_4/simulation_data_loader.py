#!/usr/bin/env python3
"""
Load and prepare simulation data for statistical analysis,
particularly comparing scores with and without player p4 involvement.
"""

import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path


def load_simulation_data(filepath):
    """
    Load simulation data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    try:
        df = pd.read_csv(filepath)
        print(f"✓ Loaded {len(df)} rows from {filepath}")
        return df
    except Exception as e:
        print(f"✗ Error loading file: {e}")
        sys.exit(1)


def prepare_data_for_analysis(df):
    """
    Prepare the data for statistical analysis by adding relevant flags
    and grouping information.
    
    Args:
        df: DataFrame with simulation data
        
    Returns:
        Tuple of (processed_df, summary_stats)
    """
    # Create a flag for whether p4 is involved in each game
    df['has_p4'] = df['player_list'].str.contains('p4', na=False)
    
    # Extract unique games (since each row represents a player in a game)
    # Group by simulation_id and run_number to get unique games
    games_df = df.groupby(['simulation_id', 'run_number']).first().reset_index()
    
    # Get summary statistics
    summary_stats = {
        'total_rows': len(df),
        'unique_simulations': df['simulation_id'].nunique(),
        'unique_games': len(games_df),
        'games_with_p4': games_df['has_p4'].sum(),
        'games_without_p4': (~games_df['has_p4']).sum(),
        'player_types': df['player_selection_type'].unique().tolist(),
        'unique_players': df['player_name'].nunique()
    }
    
    return df, games_df, summary_stats


def analyze_score_differences(df, games_df):
    """
    Analyze score differences between games with and without p4.
    
    Args:
        df: Full DataFrame with all player records
        games_df: DataFrame with unique games
        
    Returns:
        Dictionary with comparative statistics
    """
    with_p4 = games_df[games_df['has_p4']]
    without_p4 = games_df[~games_df['has_p4']]
    
    # Score columns to analyze
    score_columns = [
        'shared_score_total',
        'shared_score_importance', 
        'shared_score_coherence',
        'shared_score_freshness',
        'shared_score_nonmonotonousness'
    ]
    
    comparison = {}
    
    for col in score_columns:
        if col in games_df.columns:
            comparison[col] = {
                'with_p4': {
                    'mean': with_p4[col].mean() if len(with_p4) > 0 else np.nan,
                    'std': with_p4[col].std() if len(with_p4) > 0 else np.nan,
                    'median': with_p4[col].median() if len(with_p4) > 0 else np.nan,
                    'count': len(with_p4)
                },
                'without_p4': {
                    'mean': without_p4[col].mean() if len(without_p4) > 0 else np.nan,
                    'std': without_p4[col].std() if len(without_p4) > 0 else np.nan,
                    'median': without_p4[col].median() if len(without_p4) > 0 else np.nan,
                    'count': len(without_p4)
                }
            }
    
    # Also analyze individual player scores
    player_with_p4 = df[df['has_p4']]
    player_without_p4 = df[~df['has_p4']]
    
    player_score_cols = ['player_score_total', 'player_score_shared', 'player_score_individual']
    
    for col in player_score_cols:
        if col in df.columns:
            comparison[col] = {
                'with_p4': {
                    'mean': player_with_p4[col].mean() if len(player_with_p4) > 0 else np.nan,
                    'std': player_with_p4[col].std() if len(player_with_p4) > 0 else np.nan,
                    'median': player_with_p4[col].median() if len(player_with_p4) > 0 else np.nan,
                    'count': len(player_with_p4)
                },
                'without_p4': {
                    'mean': player_without_p4[col].mean() if len(player_without_p4) > 0 else np.nan,
                    'std': player_without_p4[col].std() if len(player_without_p4) > 0 else np.nan,
                    'median': player_without_p4[col].median() if len(player_without_p4) > 0 else np.nan,
                    'count': len(player_without_p4)
                }
            }
    
    return comparison


def analyze_single_p4_vs_prp(df):
    """
    Analyze performance of single_p4 vs single_prp games.
    Normalizes scores by conversation length for fair comparison.
    
    Args:
        df: Full DataFrame with all player records
        
    Returns:
        Dictionary with comparative analysis
    """
    # Filter for single_p4 and single_prp games
    df_single = df[df['player_selection_type'].isin(['single_p4', 'single_prp'])].copy()
    
    if len(df_single) == 0:
        return None
    
    # Calculate per-turn scores (normalize by conversation length)
    df_single['shared_score_per_turn'] = df_single['shared_score_total'] / df_single['conversation_length']
    df_single['individual_score_per_turn'] = df_single['player_score_individual'] / df_single['conversation_length']
    # Note: player_score_total is already per-turn
    
    # Separate p4 and prp data
    df_p4 = df_single[df_single['player_selection_type'] == 'single_p4']
    df_prp = df_single[df_single['player_selection_type'] == 'single_prp']
    
    # Calculate statistics for each type
    stats = {}
    
    # For single_p4
    if len(df_p4) > 0:
        stats['single_p4'] = {
            'count': len(df_p4),
            'player_score_total': {
                'mean': df_p4['player_score_total'].mean(),
                'std': df_p4['player_score_total'].std(),
                'median': df_p4['player_score_total'].median()
            },
            'shared_score_per_turn': {
                'mean': df_p4['shared_score_per_turn'].mean(),
                'std': df_p4['shared_score_per_turn'].std(),
                'median': df_p4['shared_score_per_turn'].median()
            },
            'individual_score_per_turn': {
                'mean': df_p4['individual_score_per_turn'].mean(),
                'std': df_p4['individual_score_per_turn'].std(),
                'median': df_p4['individual_score_per_turn'].median()
            },
            # Raw shared scores for context
            'shared_score_total_raw': {
                'mean': df_p4['shared_score_total'].mean(),
                'std': df_p4['shared_score_total'].std()
            },
            'avg_conversation_length': df_p4['conversation_length'].mean(),
            'avg_memory_size': df_p4['memory_size'].mean()
        }
    else:
        stats['single_p4'] = None
    
    # For single_prp (Random Pause Player)
    if len(df_prp) > 0:
        stats['single_prp'] = {
            'count': len(df_prp),
            'player_score_total': {
                'mean': df_prp['player_score_total'].mean(),
                'std': df_prp['player_score_total'].std(),
                'median': df_prp['player_score_total'].median()
            },
            'shared_score_per_turn': {
                'mean': df_prp['shared_score_per_turn'].mean(),
                'std': df_prp['shared_score_per_turn'].std(),
                'median': df_prp['shared_score_per_turn'].median()
            },
            'individual_score_per_turn': {
                'mean': df_prp['individual_score_per_turn'].mean(),
                'std': df_prp['individual_score_per_turn'].std(),
                'median': df_prp['individual_score_per_turn'].median()
            },
            # Raw shared scores for context
            'shared_score_total_raw': {
                'mean': df_prp['shared_score_total'].mean(),
                'std': df_prp['shared_score_total'].std()
            },
            'avg_conversation_length': df_prp['conversation_length'].mean(),
            'avg_memory_size': df_prp['memory_size'].mean()
        }
    else:
        stats['single_prp'] = None
    
    # Calculate improvements (p4 vs prp)
    improvements = {}
    if stats['single_p4'] and stats['single_prp']:
        improvements = {
            'player_score_total': {
                'absolute': stats['single_p4']['player_score_total']['mean'] - stats['single_prp']['player_score_total']['mean'],
                'percentage': ((stats['single_p4']['player_score_total']['mean'] - stats['single_prp']['player_score_total']['mean']) / 
                              abs(stats['single_prp']['player_score_total']['mean']) * 100) if stats['single_prp']['player_score_total']['mean'] != 0 else None
            },
            'shared_score_per_turn': {
                'absolute': stats['single_p4']['shared_score_per_turn']['mean'] - stats['single_prp']['shared_score_per_turn']['mean'],
                'percentage': ((stats['single_p4']['shared_score_per_turn']['mean'] - stats['single_prp']['shared_score_per_turn']['mean']) / 
                              abs(stats['single_prp']['shared_score_per_turn']['mean']) * 100) if stats['single_prp']['shared_score_per_turn']['mean'] != 0 else None
            },
            'individual_score_per_turn': {
                'absolute': stats['single_p4']['individual_score_per_turn']['mean'] - stats['single_prp']['individual_score_per_turn']['mean'],
                'percentage': ((stats['single_p4']['individual_score_per_turn']['mean'] - stats['single_prp']['individual_score_per_turn']['mean']) / 
                              abs(stats['single_prp']['individual_score_per_turn']['mean']) * 100) if stats['single_prp']['individual_score_per_turn']['mean'] != 0 else None
            }
        }
    
    # Group by conversation length to see if performance varies with length
    length_analysis = df_single.groupby(['player_selection_type', 'conversation_length']).agg({
        'player_score_total': ['mean', 'std', 'count'],
        'shared_score_per_turn': ['mean', 'std'],
        'individual_score_per_turn': ['mean', 'std']
    }).round(4)
    
    return {
        'stats': stats,
        'improvements': improvements,
        'length_analysis': length_analysis,
        'raw_data': df_single
    }


def print_single_p4_vs_prp_analysis(analysis):
    """
    Print formatted analysis of single_p4 vs single_prp.
    """
    if analysis is None:
        print("\nNo single_p4 or single_prp games found in the dataset.")
        return
        
    print("\n" + "="*60)
    print("SINGLE PLAYER ANALYSIS: P4 vs Random Pause Player (PRP)")
    print("="*60)
    
    stats = analysis['stats']
    
    # Print P4 stats
    if stats['single_p4']:
        p4 = stats['single_p4']
        print(f"\nSingle P4 Performance (n={p4['count']}):")
        print(f"  Average conversation length: {p4['avg_conversation_length']:.1f} turns")
        print(f"  Average memory size: {p4['avg_memory_size']:.1f}")
        print(f"\n  Per-Turn Scores (normalized by conversation length):")
        print(f"    Total Score:      {p4['player_score_total']['mean']:.4f} ± {p4['player_score_total']['std']:.4f}")
        print(f"    Shared Score:     {p4['shared_score_per_turn']['mean']:.4f} ± {p4['shared_score_per_turn']['std']:.4f}")
        print(f"    Individual Score: {p4['individual_score_per_turn']['mean']:.4f} ± {p4['individual_score_per_turn']['std']:.4f}")
        print(f"\n  Raw Shared Score (not normalized):")
        print(f"    {p4['shared_score_total_raw']['mean']:.4f} ± {p4['shared_score_total_raw']['std']:.4f}")
    else:
        print("\nNo single_p4 games found.")
    
    # Print PRP stats
    if stats['single_prp']:
        prp = stats['single_prp']
        print(f"\nRandom Pause Player Performance (n={prp['count']}):")
        print(f"  Average conversation length: {prp['avg_conversation_length']:.1f} turns")
        print(f"  Average memory size: {prp['avg_memory_size']:.1f}")
        print(f"\n  Per-Turn Scores (normalized by conversation length):")
        print(f"    Total Score:      {prp['player_score_total']['mean']:.4f} ± {prp['player_score_total']['std']:.4f}")
        print(f"    Shared Score:     {prp['shared_score_per_turn']['mean']:.4f} ± {prp['shared_score_per_turn']['std']:.4f}")
        print(f"    Individual Score: {prp['individual_score_per_turn']['mean']:.4f} ± {prp['individual_score_per_turn']['std']:.4f}")
        print(f"\n  Raw Shared Score (not normalized):")
        print(f"    {prp['shared_score_total_raw']['mean']:.4f} ± {prp['shared_score_total_raw']['std']:.4f}")
    else:
        print("\nNo single_prp games found.")
    
    # Print comparison
    improvements = analysis['improvements']
    if improvements:
        print(f"\n" + "-"*60)
        print("P4 Performance vs Random Baseline:")
        print("-"*60)
        
        for metric_name, display_name in [
            ('player_score_total', 'Total Score (per-turn)'),
            ('shared_score_per_turn', 'Shared Score (per-turn)'),
            ('individual_score_per_turn', 'Individual Score (per-turn)')
        ]:
            if metric_name in improvements:
                imp = improvements[metric_name]
                print(f"\n  {display_name}:")
                print(f"    Absolute improvement: {imp['absolute']:+.4f}")
                if imp['percentage'] is not None:
                    print(f"    Percentage improvement: {imp['percentage']:+.1f}%")
                    if imp['percentage'] > 0:
                        print(f"    → P4 performs {abs(imp['percentage']):.1f}% better")
                    elif imp['percentage'] < 0:
                        print(f"    → P4 performs {abs(imp['percentage']):.1f}% worse")
                    else:
                        print(f"    → Performance is equal")
    
    # Print CSV-friendly format for plotting
    if stats['single_p4'] and stats['single_prp']:
        print(f"\n" + "-"*60)
        print("Data for Plotting (CSV format):")
        print("-"*60)
        print("metric,p4_mean,p4_std,prp_mean,prp_std,improvement_abs,improvement_pct")
        
        for metric_name, display_name in [
            ('player_score_total', 'total_per_turn'),
            ('shared_score_per_turn', 'shared_per_turn'),
            ('individual_score_per_turn', 'individual_per_turn')
        ]:
            p4_mean = stats['single_p4'][metric_name]['mean']
            p4_std = stats['single_p4'][metric_name]['std']
            prp_mean = stats['single_prp'][metric_name]['mean']
            prp_std = stats['single_prp'][metric_name]['std']
            imp_abs = improvements[metric_name]['absolute']
            imp_pct = improvements[metric_name]['percentage'] if improvements[metric_name]['percentage'] else 0
            
            print(f"{display_name},{p4_mean:.4f},{p4_std:.4f},{prp_mean:.4f},{prp_std:.4f},{imp_abs:.4f},{imp_pct:.1f}")


def analyze_1v1_with_p4(df):
    """
    Analyze 1v1 games where p4 plays against other players.
    
    Args:
        df: Full DataFrame with all player records
        
    Returns:
        Dictionary with 1v1 analysis results
    """
    # Filter for 1v1 games with p4
    df_1v1 = df[df['player_selection_type'] == '1v1'].copy()
    df_1v1_with_p4 = df_1v1[df_1v1['player_list'].str.contains('p4', na=False)]
    
    if len(df_1v1_with_p4) == 0:
        return None
    
    # Get unique games
    games = df_1v1_with_p4.groupby(['simulation_id', 'run_number']).agg({
        'player_list': 'first',
        'conversation_length': 'first',
        'num_subjects': 'first',
        'memory_size': 'first'
    }).reset_index()
    
    # For each game, get p4's performance vs opponent
    results = []
    
    for _, game in games.iterrows():
        game_data = df_1v1_with_p4[
            (df_1v1_with_p4['simulation_id'] == game['simulation_id']) & 
            (df_1v1_with_p4['run_number'] == game['run_number'])
        ]
        
        # Get p4's row and opponent's row
        p4_row = game_data[game_data['player_name'].str.contains('Player4|p4', case=False, na=False)]
        opponent_row = game_data[~game_data['player_name'].str.contains('Player4|p4', case=False, na=False)]
        
        if len(p4_row) == 1 and len(opponent_row) == 1:
            p4_row = p4_row.iloc[0]
            opponent_row = opponent_row.iloc[0]
            
            # Extract opponent name from player_list
            players = game['player_list'].split(',')
            opponent_name = [p for p in players if p != 'p4'][0] if len(players) == 2 else 'unknown'
            
            results.append({
                'game_id': f"{game['simulation_id']}_{game['run_number']}",
                'opponent': opponent_name,
                'opponent_full_name': opponent_row['player_name'],
                'conversation_length': game['conversation_length'],
                'memory_size': game['memory_size'],
                # P4 scores
                'p4_score_total': p4_row['player_score_total'],
                'p4_score_shared': p4_row['player_score_shared'],
                'p4_score_individual': p4_row['player_score_individual'],
                # Opponent scores  
                'opponent_score_total': opponent_row['player_score_total'],
                'opponent_score_shared': opponent_row['player_score_shared'],
                'opponent_score_individual': opponent_row['player_score_individual'],
                # Differences (p4 - opponent)
                'diff_total': p4_row['player_score_total'] - opponent_row['player_score_total'],
                'diff_shared': p4_row['player_score_shared'] - opponent_row['player_score_shared'],
                'diff_individual': p4_row['player_score_individual'] - opponent_row['player_score_individual'],
                # Shared game scores
                'shared_score_total': p4_row['shared_score_total'],
                'shared_score_importance': p4_row['shared_score_importance'],
                'shared_score_coherence': p4_row['shared_score_coherence'],
                'shared_score_freshness': p4_row['shared_score_freshness'],
                'shared_score_nonmonotonousness': p4_row['shared_score_nonmonotonousness']
            })
    
    if not results:
        return None
        
    results_df = pd.DataFrame(results)
    
    # Calculate win/loss/tie statistics
    wins = (results_df['diff_total'] > 0).sum()
    losses = (results_df['diff_total'] < 0).sum()
    ties = (results_df['diff_total'] == 0).sum()
    
    # Group by opponent to see performance against each
    by_opponent = results_df.groupby('opponent').agg({
        'diff_total': ['mean', 'std', 'count'],
        'p4_score_total': ['mean', 'std'],
        'opponent_score_total': ['mean', 'std'],
        'p4_score_individual': ['mean', 'std'],
        'opponent_score_individual': ['mean', 'std'],
        'p4_score_shared': ['mean', 'std'],
        'opponent_score_shared': ['mean', 'std']
    }).round(4)
    
    summary = {
        'total_games': len(results_df),
        'wins': wins,
        'losses': losses,
        'ties': ties,
        'win_rate': wins / len(results_df) if len(results_df) > 0 else 0,
        'avg_p4_score': results_df['p4_score_total'].mean(),
        'avg_opponent_score': results_df['opponent_score_total'].mean(),
        'avg_score_diff': results_df['diff_total'].mean(),
        'unique_opponents': results_df['opponent'].nunique()
    }
    
    return {
        'summary': summary,
        'by_opponent': by_opponent,
        'detailed_results': results_df
    }


def print_1v1_analysis(analysis_1v1):
    """
    Print formatted 1v1 analysis results.
    """
    if analysis_1v1 is None:
        print("\nNo 1v1 games with p4 found in the dataset.")
        return
    
    print("\n" + "="*60)
    print("1v1 HEAD-TO-HEAD ANALYSIS: P4 vs OTHER PLAYERS")
    print("="*60)
    
    summary = analysis_1v1['summary']
    print(f"\nOverall Performance:")
    print(f"  Total 1v1 Games: {summary['total_games']}")
    print(f"  Wins: {summary['wins']} | Losses: {summary['losses']} | Ties: {summary['ties']}")
    print(f"  Win Rate: {summary['win_rate']:.1%}")
    print(f"  Unique Opponents: {summary['unique_opponents']}")
    print(f"\n  Average Scores:")
    print(f"    P4:       {summary['avg_p4_score']:.4f}")
    print(f"    Opponents: {summary['avg_opponent_score']:.4f}")
    print(f"    Difference: {summary['avg_score_diff']:+.4f}")
    
    print(f"\n" + "-"*60)
    print("Performance by Opponent (mean ± std):")
    print("-"*60)
    
    by_opponent = analysis_1v1['by_opponent']
    
    # Create a cleaner display with standard deviations
    for opponent in by_opponent.index:
        games_played = int(by_opponent.loc[opponent, ('diff_total', 'count')])
        
        # Score totals
        p4_avg = by_opponent.loc[opponent, ('p4_score_total', 'mean')]
        p4_std = by_opponent.loc[opponent, ('p4_score_total', 'std')]
        opp_avg = by_opponent.loc[opponent, ('opponent_score_total', 'mean')]
        opp_std = by_opponent.loc[opponent, ('opponent_score_total', 'std')]
        
        # Score differences
        avg_diff = by_opponent.loc[opponent, ('diff_total', 'mean')]
        std_diff = by_opponent.loc[opponent, ('diff_total', 'std')]
        
        # Individual scores
        p4_ind_avg = by_opponent.loc[opponent, ('p4_score_individual', 'mean')]
        p4_ind_std = by_opponent.loc[opponent, ('p4_score_individual', 'std')]
        opp_ind_avg = by_opponent.loc[opponent, ('opponent_score_individual', 'mean')]
        opp_ind_std = by_opponent.loc[opponent, ('opponent_score_individual', 'std')]
        
        # Shared scores
        p4_shared_avg = by_opponent.loc[opponent, ('p4_score_shared', 'mean')]
        p4_shared_std = by_opponent.loc[opponent, ('p4_score_shared', 'std')]
        opp_shared_avg = by_opponent.loc[opponent, ('opponent_score_shared', 'mean')]
        opp_shared_std = by_opponent.loc[opponent, ('opponent_score_shared', 'std')]
        
        print(f"\n  vs {opponent} ({games_played} game{'s' if games_played > 1 else ''}):")
        print(f"    Total Scores:")
        print(f"      P4:         {p4_avg:.4f} ± {p4_std:.4f}")
        print(f"      {opponent}:      {opp_avg:.4f} ± {opp_std:.4f}")
        print(f"      Difference: {avg_diff:+.4f} ± {std_diff:.4f}")
        
        if games_played > 1:  # Only show component scores if multiple games
            print(f"    Individual Scores:")
            print(f"      P4:         {p4_ind_avg:.4f} ± {p4_ind_std:.4f}")
            print(f"      {opponent}:      {opp_ind_avg:.4f} ± {opp_ind_std:.4f}")
            print(f"    Shared Scores:")
            print(f"      P4:         {p4_shared_avg:.4f} ± {p4_shared_std:.4f}")
            print(f"      {opponent}:      {opp_shared_avg:.4f} ± {opp_shared_std:.4f}")
        
        if avg_diff > 0:
            print(f"    Result: P4 wins on average")
        elif avg_diff < 0:
            print(f"    Result: {opponent} wins on average")
        else:
            print(f"    Result: Tied on average")
    
    # Export-friendly format for plotting
    print(f"\n" + "-"*60)
    print("Data for Plotting (CSV-friendly format):")
    print("-"*60)
    print("opponent,games,p4_mean,p4_std,opp_mean,opp_std,diff_mean,diff_std")
    for opponent in by_opponent.index:
        games = int(by_opponent.loc[opponent, ('diff_total', 'count')])
        p4_mean = by_opponent.loc[opponent, ('p4_score_total', 'mean')]
        p4_std = by_opponent.loc[opponent, ('p4_score_total', 'std')]
        opp_mean = by_opponent.loc[opponent, ('opponent_score_total', 'mean')]
        opp_std = by_opponent.loc[opponent, ('opponent_score_total', 'std')]
        diff_mean = by_opponent.loc[opponent, ('diff_total', 'mean')]
        diff_std = by_opponent.loc[opponent, ('diff_total', 'std')]
        print(f"{opponent},{games},{p4_mean:.4f},{p4_std:.4f},{opp_mean:.4f},{opp_std:.4f},{diff_mean:.4f},{diff_std:.4f}")
    
    # Show detailed game results if not too many
    detailed = analysis_1v1['detailed_results']
    if len(detailed) <= 20:  # Only show detailed results for small datasets
        print(f"\n" + "-"*60)
        print("Detailed Game Results:")
        print("-"*60)
        for _, game in detailed.iterrows():
            result = "WIN" if game['diff_total'] > 0 else ("LOSS" if game['diff_total'] < 0 else "TIE")
            print(f"  vs {game['opponent']}: P4={game['p4_score_total']:.3f}, "
                  f"Opp={game['opponent_score_total']:.3f}, "
                  f"Diff={game['diff_total']:+.3f} [{result}]")


def print_summary(summary_stats, comparison):
    """
    Print a formatted summary of the data and comparisons.
    """
    print("\n" + "="*60)
    print("DATA SUMMARY")
    print("="*60)
    
    for key, value in summary_stats.items():
        print(f"{key:25s}: {value}")
    
    print("\n" + "="*60)
    print("SCORE COMPARISONS (Games with p4 vs without p4)")
    print("="*60)
    
    for metric, stats in comparison.items():
        print(f"\n{metric}:")
        if not np.isnan(stats['with_p4']['mean']):
            print(f"  With p4 (n={stats['with_p4']['count']:,}):")
            print(f"    Mean:   {stats['with_p4']['mean']:.4f}")
            print(f"    Std:    {stats['with_p4']['std']:.4f}")
            print(f"    Median: {stats['with_p4']['median']:.4f}")
        else:
            print(f"  With p4: No data available")
            
        if not np.isnan(stats['without_p4']['mean']):
            print(f"  Without p4 (n={stats['without_p4']['count']:,}):")
            print(f"    Mean:   {stats['without_p4']['mean']:.4f}")
            print(f"    Std:    {stats['without_p4']['std']:.4f}")
            print(f"    Median: {stats['without_p4']['median']:.4f}")
        else:
            print(f"  Without p4: No data available")
            
        if not np.isnan(stats['with_p4']['mean']) and not np.isnan(stats['without_p4']['mean']):
            diff = stats['with_p4']['mean'] - stats['without_p4']['mean']
            print(f"  Difference (with - without): {diff:+.4f}")


def analyze_all_players_performance(df):
    """
    Analyze normalized per-turn performance of all players (p1-p10 and prp) 
    across all games they participate in.
    
    Args:
        df: Full DataFrame with all player records
        
    Returns:
        Dictionary with per-player performance statistics
    """
    # Create a copy to work with
    df_work = df.copy()
    
    # Calculate per-turn scores (normalize by conversation length)
    df_work['shared_score_per_turn'] = df_work['shared_score_total'] / df_work['conversation_length']
    df_work['individual_score_per_turn'] = df_work['player_score_individual'] / df_work['conversation_length']
    # Note: player_score_total is already per-turn
    
    # Extract player identifier from player_name or player_list
    def get_player_id(row):
        """Extract standardized player ID from player name."""
        name = str(row['player_name']).lower()
        
        # Check for specific player patterns
        if 'randompauseplayer' in name.replace(' ', '').lower():
            return 'prp'
        elif 'player10' in name:
            return 'p10'
        elif 'player1' in name:
            return 'p1'
        elif 'player2' in name:
            return 'p2'
        elif 'player3' in name:
            return 'p3'
        elif 'player4' in name:
            return 'p4'
        elif 'player5' in name:
            return 'p5'
        elif 'player6' in name:
            return 'p6'
        elif 'player7' in name:
            return 'p7'
        elif 'player8' in name:
            return 'p8'
        elif 'player9' in name:
            return 'p9'
        else:
            # Try to extract from patterns like "p1", "p2", etc.
            import re
            match = re.search(r'p(\d+)', name)
            if match:
                return f'p{match.group(1)}'
            return 'unknown'
    
    df_work['player_id'] = df_work.apply(get_player_id, axis=1)
    
    # Filter out unknown players
    df_work = df_work[df_work['player_id'] != 'unknown']
    
    # Calculate statistics for each player
    player_stats = {}
    all_players = sorted(df_work['player_id'].unique())
    
    for player in all_players:
        player_data = df_work[df_work['player_id'] == player]
        
        if len(player_data) > 0:
            player_stats[player] = {
                'count': len(player_data),
                'unique_games': player_data.groupby(['simulation_id', 'run_number']).ngroups,
                'player_score_total': {
                    'mean': player_data['player_score_total'].mean(),
                    'std': player_data['player_score_total'].std(),
                    'median': player_data['player_score_total'].median(),
                    'min': player_data['player_score_total'].min(),
                    'max': player_data['player_score_total'].max()
                },
                'shared_score_per_turn': {
                    'mean': player_data['shared_score_per_turn'].mean(),
                    'std': player_data['shared_score_per_turn'].std(),
                    'median': player_data['shared_score_per_turn'].median()
                },
                'individual_score_per_turn': {
                    'mean': player_data['individual_score_per_turn'].mean(),
                    'std': player_data['individual_score_per_turn'].std(),
                    'median': player_data['individual_score_per_turn'].median()
                },
                'avg_conversation_length': player_data['conversation_length'].mean(),
                'game_types': player_data['player_selection_type'].value_counts().to_dict()
            }
    
    # Calculate rankings
    rankings = {}
    for metric in ['player_score_total', 'shared_score_per_turn', 'individual_score_per_turn']:
        player_means = [(player, stats[metric]['mean']) 
                       for player, stats in player_stats.items()]
        player_means.sort(key=lambda x: x[1], reverse=True)
        rankings[metric] = player_means
    
    return {
        'player_stats': player_stats,
        'rankings': rankings,
        'all_players': all_players
    }


def print_all_players_analysis(analysis):
    """
    Print formatted analysis of all players' performance.
    """
    if analysis is None:
        print("\nNo player data found in the dataset.")
        return
    
    print("\n" + "="*60)
    print("ALL PLAYERS PERFORMANCE COMPARISON")
    print("="*60)
    
    player_stats = analysis['player_stats']
    rankings = analysis['rankings']
    
    # Print overall rankings
    print("\n" + "-"*60)
    print("Player Rankings by Total Score (per-turn):")
    print("-"*60)
    
    for rank, (player, score) in enumerate(rankings['player_score_total'], 1):
        games = player_stats[player]['unique_games']
        std = player_stats[player]['player_score_total']['std']
        print(f"  {rank:2d}. {player.upper():5s}: {score:.4f} ± {std:.4f} ({games} games)")
    
    # Find P4's rank
    p4_rank = next((i+1 for i, (p, _) in enumerate(rankings['player_score_total']) if p == 'p4'), None)
    if p4_rank:
        total_players = len(rankings['player_score_total'])
        print(f"\n  → P4 ranks {p4_rank}/{total_players} among all players")
    
    # Print detailed comparison table
    print("\n" + "-"*60)
    print("Detailed Performance Metrics (All Players):")
    print("-"*60)
    print(f"{'Player':6s} {'Games':>6s} {'Total/Turn':>12s} {'Shared/Turn':>12s} {'Indiv/Turn':>12s}")
    print(f"{'':6s} {'':>6s} {'(mean±std)':>12s} {'(mean±std)':>12s} {'(mean±std)':>12s}")
    print("-"*60)
    
    # Sort players for consistent display (p1-p10, then prp)
    sorted_players = []
    for i in range(1, 11):
        if f'p{i}' in player_stats:
            sorted_players.append(f'p{i}')
    if 'prp' in player_stats:
        sorted_players.append('prp')
    
    for player in sorted_players:
        if player in player_stats:
            stats = player_stats[player]
            games = stats['unique_games']
            
            total_mean = stats['player_score_total']['mean']
            total_std = stats['player_score_total']['std']
            shared_mean = stats['shared_score_per_turn']['mean']
            shared_std = stats['shared_score_per_turn']['std']
            indiv_mean = stats['individual_score_per_turn']['mean']
            indiv_std = stats['individual_score_per_turn']['std']
            
            # Highlight P4
            if player == 'p4':
                print(f"→ {player.upper():5s} {games:>6d} "
                      f"{total_mean:>5.3f}±{total_std:<5.3f} "
                      f"{shared_mean:>5.3f}±{shared_std:<5.3f} "
                      f"{indiv_mean:>5.3f}±{indiv_std:<5.3f} ←")
            else:
                print(f"  {player.upper():5s} {games:>6d} "
                      f"{total_mean:>5.3f}±{total_std:<5.3f} "
                      f"{shared_mean:>5.3f}±{shared_std:<5.3f} "
                      f"{indiv_mean:>5.3f}±{indiv_std:<5.3f}")
    
    # Performance comparison vs PRP
    if 'p4' in player_stats and 'prp' in player_stats:
        p4_total = player_stats['p4']['player_score_total']['mean']
        prp_total = player_stats['prp']['player_score_total']['mean']
        improvement = ((p4_total - prp_total) / abs(prp_total) * 100) if prp_total != 0 else 0
        
        print("\n" + "-"*60)
        print("P4 vs Random Baseline (PRP):")
        print("-"*60)
        print(f"  P4 total score:  {p4_total:.4f}")
        print(f"  PRP total score: {prp_total:.4f}")
        print(f"  Improvement:     {improvement:+.1f}%")
    
    # CSV output for plotting
    print("\n" + "-"*60)
    print("Data for Plotting (CSV format):")
    print("-"*60)
    print("player,games,total_mean,total_std,shared_mean,shared_std,indiv_mean,indiv_std")
    
    for player in sorted_players:
        if player in player_stats:
            stats = player_stats[player]
            print(f"{player},{stats['unique_games']},"
                  f"{stats['player_score_total']['mean']:.4f},{stats['player_score_total']['std']:.4f},"
                  f"{stats['shared_score_per_turn']['mean']:.4f},{stats['shared_score_per_turn']['std']:.4f},"
                  f"{stats['individual_score_per_turn']['mean']:.4f},{stats['individual_score_per_turn']['std']:.4f}")


def analyze_10_unique_vs_9_omit(df):
    """
    Analyze 10_unique (with p4) vs 9_omit_p4 (without p4) games.
    This shows if adding p4 as the 10th player improves scores.
    
    Args:
        df: Full DataFrame with all player records
        
    Returns:
        Dictionary with comparative analysis
    """
    # Filter for the specific player selection types
    df_10_unique = df[df['player_selection_type'] == '10_unique'].copy()
    df_9_omit = df[df['player_selection_type'] == '9_omit_p4'].copy()
    
    if len(df_10_unique) == 0 and len(df_9_omit) == 0:
        return None
    
    # Get unique games for each type (since multiple rows per game)
    games_10_unique = df_10_unique.groupby(['simulation_id', 'run_number']).first().reset_index()
    games_9_omit = df_9_omit.groupby(['simulation_id', 'run_number']).first().reset_index()
    
    stats = {}
    
    # Calculate statistics for 10_unique (with p4)
    if len(games_10_unique) > 0:
        stats['10_unique'] = {
            'num_games': len(games_10_unique),
            'num_players_per_game': df_10_unique.groupby(['simulation_id', 'run_number']).size().mean(),
            'shared_scores': {
                'total': {
                    'mean': games_10_unique['shared_score_total'].mean(),
                    'std': games_10_unique['shared_score_total'].std(),
                    'median': games_10_unique['shared_score_total'].median()
                },
                'importance': {
                    'mean': games_10_unique['shared_score_importance'].mean(),
                    'std': games_10_unique['shared_score_importance'].std()
                },
                'coherence': {
                    'mean': games_10_unique['shared_score_coherence'].mean(),
                    'std': games_10_unique['shared_score_coherence'].std()
                },
                'freshness': {
                    'mean': games_10_unique['shared_score_freshness'].mean(),
                    'std': games_10_unique['shared_score_freshness'].std()
                },
                'nonmonotonousness': {
                    'mean': games_10_unique['shared_score_nonmonotonousness'].mean(),
                    'std': games_10_unique['shared_score_nonmonotonousness'].std()
                }
            },
            'player_scores': {
                'total': {
                    'mean': df_10_unique['player_score_total'].mean(),
                    'std': df_10_unique['player_score_total'].std()
                },
                'individual': {
                    'mean': df_10_unique['player_score_individual'].mean(),
                    'std': df_10_unique['player_score_individual'].std()
                },
                'shared': {
                    'mean': df_10_unique['player_score_shared'].mean(),
                    'std': df_10_unique['player_score_shared'].std()
                }
            },
            'avg_conversation_length': games_10_unique['conversation_length'].mean(),
            'avg_memory_size': games_10_unique['memory_size'].mean()
        }
        
        # Get P4's specific performance in 10_unique games
        p4_in_10 = df_10_unique[df_10_unique['player_name'].str.contains('Player4|p4', case=False, na=False)]
        if len(p4_in_10) > 0:
            stats['10_unique']['p4_specific'] = {
                'total': p4_in_10['player_score_total'].mean(),
                'individual': p4_in_10['player_score_individual'].mean(),
                'shared': p4_in_10['player_score_shared'].mean()
            }
    else:
        stats['10_unique'] = None
    
    # Calculate statistics for 9_omit_p4 (without p4)
    if len(games_9_omit) > 0:
        stats['9_omit_p4'] = {
            'num_games': len(games_9_omit),
            'num_players_per_game': df_9_omit.groupby(['simulation_id', 'run_number']).size().mean(),
            'shared_scores': {
                'total': {
                    'mean': games_9_omit['shared_score_total'].mean(),
                    'std': games_9_omit['shared_score_total'].std(),
                    'median': games_9_omit['shared_score_total'].median()
                },
                'importance': {
                    'mean': games_9_omit['shared_score_importance'].mean(),
                    'std': games_9_omit['shared_score_importance'].std()
                },
                'coherence': {
                    'mean': games_9_omit['shared_score_coherence'].mean(),
                    'std': games_9_omit['shared_score_coherence'].std()
                },
                'freshness': {
                    'mean': games_9_omit['shared_score_freshness'].mean(),
                    'std': games_9_omit['shared_score_freshness'].std()
                },
                'nonmonotonousness': {
                    'mean': games_9_omit['shared_score_nonmonotonousness'].mean(),
                    'std': games_9_omit['shared_score_nonmonotonousness'].std()
                }
            },
            'player_scores': {
                'total': {
                    'mean': df_9_omit['player_score_total'].mean(),
                    'std': df_9_omit['player_score_total'].std()
                },
                'individual': {
                    'mean': df_9_omit['player_score_individual'].mean(),
                    'std': df_9_omit['player_score_individual'].std()
                },
                'shared': {
                    'mean': df_9_omit['player_score_shared'].mean(),
                    'std': df_9_omit['player_score_shared'].std()
                }
            },
            'avg_conversation_length': games_9_omit['conversation_length'].mean(),
            'avg_memory_size': games_9_omit['memory_size'].mean()
        }
    else:
        stats['9_omit_p4'] = None
    
    # Calculate improvements (10_unique vs 9_omit_p4)
    improvements = {}
    if stats['10_unique'] and stats['9_omit_p4']:
        # Shared score improvements
        for score_type in ['total', 'importance', 'coherence', 'freshness', 'nonmonotonousness']:
            if score_type == 'total':
                mean_10 = stats['10_unique']['shared_scores'][score_type]['mean']
                mean_9 = stats['9_omit_p4']['shared_scores'][score_type]['mean']
            else:
                mean_10 = stats['10_unique']['shared_scores'][score_type]['mean']
                mean_9 = stats['9_omit_p4']['shared_scores'][score_type]['mean']
            
            improvements[f'shared_{score_type}'] = {
                'absolute': mean_10 - mean_9,
                'percentage': ((mean_10 - mean_9) / abs(mean_9) * 100) if mean_9 != 0 else None
            }
        
        # Player score improvements (average per player)
        for score_type in ['total', 'individual', 'shared']:
            mean_10 = stats['10_unique']['player_scores'][score_type]['mean']
            mean_9 = stats['9_omit_p4']['player_scores'][score_type]['mean']
            
            improvements[f'player_{score_type}'] = {
                'absolute': mean_10 - mean_9,
                'percentage': ((mean_10 - mean_9) / abs(mean_9) * 100) if mean_9 != 0 else None
            }
    
    return {
        'stats': stats,
        'improvements': improvements
    }


def print_10_unique_vs_9_omit_analysis(analysis):
    """
    Print formatted analysis of 10_unique vs 9_omit_p4.
    """
    if analysis is None:
        print("\nNo 10_unique or 9_omit_p4 games found in the dataset.")
        return
        
    print("\n" + "="*60)
    print("GROUP COMPARISON: 10 Players (with P4) vs 9 Players (without P4)")
    print("="*60)
    
    stats = analysis['stats']
    
    # Print 10_unique stats
    if stats['10_unique']:
        s = stats['10_unique']
        print(f"\n10 Unique Players (INCLUDING P4) - {s['num_games']} games:")
        print(f"  Players per game: {s['num_players_per_game']:.1f}")
        print(f"  Avg conversation length: {s['avg_conversation_length']:.1f}")
        print(f"  Avg memory size: {s['avg_memory_size']:.1f}")
        
        print(f"\n  Shared Scores (game-level):")
        print(f"    Total:             {s['shared_scores']['total']['mean']:.4f} ± {s['shared_scores']['total']['std']:.4f}")
        print(f"    Importance:        {s['shared_scores']['importance']['mean']:.4f} ± {s['shared_scores']['importance']['std']:.4f}")
        print(f"    Coherence:         {s['shared_scores']['coherence']['mean']:.4f} ± {s['shared_scores']['coherence']['std']:.4f}")
        print(f"    Freshness:         {s['shared_scores']['freshness']['mean']:.4f} ± {s['shared_scores']['freshness']['std']:.4f}")
        print(f"    Non-monotonousness: {s['shared_scores']['nonmonotonousness']['mean']:.4f} ± {s['shared_scores']['nonmonotonousness']['std']:.4f}")
        
        print(f"\n  Average Player Scores:")
        print(f"    Total:      {s['player_scores']['total']['mean']:.4f} ± {s['player_scores']['total']['std']:.4f}")
        print(f"    Individual: {s['player_scores']['individual']['mean']:.4f} ± {s['player_scores']['individual']['std']:.4f}")
        print(f"    Shared:     {s['player_scores']['shared']['mean']:.4f} ± {s['player_scores']['shared']['std']:.4f}")
        
        if 'p4_specific' in s:
            print(f"\n  P4's Specific Performance:")
            print(f"    Total:      {s['p4_specific']['total']:.4f}")
            print(f"    Individual: {s['p4_specific']['individual']:.4f}")
            print(f"    Shared:     {s['p4_specific']['shared']:.4f}")
    else:
        print("\nNo 10_unique games found.")
    
    # Print 9_omit_p4 stats
    if stats['9_omit_p4']:
        s = stats['9_omit_p4']
        print(f"\n9 Players (EXCLUDING P4) - {s['num_games']} games:")
        print(f"  Players per game: {s['num_players_per_game']:.1f}")
        print(f"  Avg conversation length: {s['avg_conversation_length']:.1f}")
        print(f"  Avg memory size: {s['avg_memory_size']:.1f}")
        
        print(f"\n  Shared Scores (game-level):")
        print(f"    Total:             {s['shared_scores']['total']['mean']:.4f} ± {s['shared_scores']['total']['std']:.4f}")
        print(f"    Importance:        {s['shared_scores']['importance']['mean']:.4f} ± {s['shared_scores']['importance']['std']:.4f}")
        print(f"    Coherence:         {s['shared_scores']['coherence']['mean']:.4f} ± {s['shared_scores']['coherence']['std']:.4f}")
        print(f"    Freshness:         {s['shared_scores']['freshness']['mean']:.4f} ± {s['shared_scores']['freshness']['std']:.4f}")
        print(f"    Non-monotonousness: {s['shared_scores']['nonmonotonousness']['mean']:.4f} ± {s['shared_scores']['nonmonotonousness']['std']:.4f}")
        
        print(f"\n  Average Player Scores:")
        print(f"    Total:      {s['player_scores']['total']['mean']:.4f} ± {s['player_scores']['total']['std']:.4f}")
        print(f"    Individual: {s['player_scores']['individual']['mean']:.4f} ± {s['player_scores']['individual']['std']:.4f}")
        print(f"    Shared:     {s['player_scores']['shared']['mean']:.4f} ± {s['player_scores']['shared']['std']:.4f}")
    else:
        print("\nNo 9_omit_p4 games found.")
    
    # Print comparison
    improvements = analysis['improvements']
    if improvements:
        print(f"\n" + "-"*60)
        print("Impact of Adding P4 (10_unique vs 9_omit_p4):")
        print("-"*60)
        
        print("\n  Shared Score Changes:")
        for metric in ['total', 'importance', 'coherence', 'freshness', 'nonmonotonousness']:
            key = f'shared_{metric}'
            if key in improvements:
                imp = improvements[key]
                print(f"    {metric.capitalize():18s}: {imp['absolute']:+.4f}", end='')
                if imp['percentage'] is not None:
                    print(f" ({imp['percentage']:+.1f}%)", end='')
                    if imp['absolute'] > 0:
                        print(" ↑ IMPROVED")
                    elif imp['absolute'] < 0:
                        print(" ↓ DECREASED")
                    else:
                        print(" → NO CHANGE")
                else:
                    print()
        
        print("\n  Average Player Score Changes:")
        for metric in ['total', 'individual', 'shared']:
            key = f'player_{metric}'
            if key in improvements:
                imp = improvements[key]
                print(f"    {metric.capitalize():18s}: {imp['absolute']:+.4f}", end='')
                if imp['percentage'] is not None:
                    print(f" ({imp['percentage']:+.1f}%)", end='')
                    if imp['absolute'] > 0:
                        print(" ↑ IMPROVED")
                    elif imp['absolute'] < 0:
                        print(" ↓ DECREASED")
                    else:
                        print(" → NO CHANGE")
                else:
                    print()
    
    # Print CSV-friendly format for plotting
    if stats['10_unique'] and stats['9_omit_p4']:
        print(f"\n" + "-"*60)
        print("Data for Plotting (CSV format):")
        print("-"*60)
        print("metric,with_p4_mean,with_p4_std,without_p4_mean,without_p4_std,improvement_abs,improvement_pct")
        
        metrics_to_export = [
            ('shared_total', 'shared_scores', 'total'),
            ('shared_importance', 'shared_scores', 'importance'),
            ('shared_coherence', 'shared_scores', 'coherence'),
            ('player_total', 'player_scores', 'total'),
            ('player_individual', 'player_scores', 'individual')
        ]
        
        for display_name, category, metric in metrics_to_export:
            with_mean = stats['10_unique'][category][metric]['mean']
            with_std = stats['10_unique'][category][metric]['std']
            without_mean = stats['9_omit_p4'][category][metric]['mean']
            without_std = stats['9_omit_p4'][category][metric]['std']
            
            if category == 'shared_scores':
                imp_key = f'shared_{metric}'
            else:
                imp_key = f'player_{metric}'
            
            imp_abs = improvements[imp_key]['absolute'] if imp_key in improvements else 0
            imp_pct = improvements[imp_key]['percentage'] if imp_key in improvements and improvements[imp_key]['percentage'] else 0
            
            print(f"{display_name},{with_mean:.4f},{with_std:.4f},{without_mean:.4f},{without_std:.4f},{imp_abs:.4f},{imp_pct:.1f}")


def main():
    """Main function to run the analysis."""
    
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
        sys.exit(1)
    
    filepath = Path(sys.argv[1])
    
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found")
        sys.exit(1)
    
    # Load the data
    df = load_simulation_data(filepath)
    
    # Prepare for analysis
    df, games_df, summary_stats = prepare_data_for_analysis(df)
    
    # Analyze score differences
    comparison = analyze_score_differences(df, games_df)
    
    # Print summary
    print_summary(summary_stats, comparison)
    
    # Analyze 1v1 games with p4
    analysis_1v1 = analyze_1v1_with_p4(df)
    print_1v1_analysis(analysis_1v1)
    
    # Analyze single_p4 vs single_prp
    analysis_single = analyze_single_p4_vs_prp(df)
    print_single_p4_vs_prp_analysis(analysis_single)
    
    # Analyze 10_unique vs 9_omit_p4
    analysis_group = analyze_10_unique_vs_9_omit(df)
    print_10_unique_vs_9_omit_analysis(analysis_group)
    
    # Analyze all players performance
    analysis_all_players = analyze_all_players_performance(df)
    print_all_players_analysis(analysis_all_players)
    
    # Return the dataframes for further analysis if needed
    return df, games_df, comparison, analysis_1v1, analysis_single, analysis_group, analysis_all_players


if __name__ == "__main__":
    df, games_df, comparison, analysis_1v1, analysis_single, analysis_group, analysis_all_players = main()
    
    # The data is now loaded and ready for further statistical analysis
    # You can access:
    # - df: Full dataframe with all player records and 'has_p4' flag
    # - games_df: Unique games with aggregated stats
    # - comparison: Dictionary with comparative statistics
    # - analysis_1v1: Dictionary with 1v1 head-to-head analysis
    # - analysis_single: Dictionary with single_p4 vs single_prp analysis
    # - analysis_group: Dictionary with 10_unique vs 9_omit_p4 analysis
    # - analysis_all_players: Dictionary with all players performance comparison