#!/usr/bin/env python3
"""
Create plots from simulation analysis results.
Run this after running the main analysis script.
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10

# Define consistent color scheme
COLORS = {
    'p4': '#90EE90',           # Light green for P4
    'with_p4': '#90EE90',       # Light green when P4 is involved
    'without_p4': '#FFB6C1',    # Light pink/coral for without P4
    'opponent': '#87CEEB',      # Sky blue for opponents
    'random': '#D3D3D3',        # Light gray for random player
    'secondary': '#DDA0DD',     # Plum for secondary data
    'accent': '#F0E68C'         # Khaki for accents
}


def ensure_output_dir(output_dir):
    """Create output directory if it doesn't exist."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def plot_scores_with_without_p4(comparison, output_path):
    """
    Plot 1: Player total scores with and without p4 (mean and std).
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Prepare data for plotting
    metrics = ['player_score_total', 'player_score_shared', 'player_score_individual']
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    with_p4_means = []
    with_p4_stds = []
    without_p4_means = []
    without_p4_stds = []
    
    for metric in metrics:
        if metric in comparison:
            with_p4_means.append(comparison[metric]['with_p4']['mean'])
            with_p4_stds.append(comparison[metric]['with_p4']['std'])
            without_p4_means.append(comparison[metric]['without_p4']['mean'])
            without_p4_stds.append(comparison[metric]['without_p4']['std'])
        else:
            with_p4_means.append(0)
            with_p4_stds.append(0)
            without_p4_means.append(0)
            without_p4_stds.append(0)
    
    # Create bars with consistent colors
    bars1 = ax.bar(x_pos - width/2, with_p4_means, width, 
                   yerr=with_p4_stds, label='With P4', 
                   capsize=5, color=COLORS['with_p4'], alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, without_p4_means, width,
                   yerr=without_p4_stds, label='Without P4',
                   capsize=5, color=COLORS['without_p4'], alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Score Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score Value', fontsize=12, fontweight='bold')
    ax.set_title('Player Scores: With vs Without P4', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Total', 'Shared', 'Individual'])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    save_path = output_path / 'scores_with_without_p4.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_p4_vs_random(analysis_single, output_path):
    """
    Plot 2: P4 vs Random Pause Player - normalized per-turn scores.
    """
    if analysis_single is None or analysis_single['stats']['single_p4'] is None or analysis_single['stats']['single_prp'] is None:
        print("⚠ Skipping P4 vs Random plot - insufficient data")
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    stats = analysis_single['stats']
    
    # Prepare data
    metrics = ['player_score_total', 'shared_score_per_turn', 'individual_score_per_turn']
    metric_labels = ['Total (per-turn)', 'Shared (per-turn)', 'Individual (per-turn)']
    
    x_pos = np.arange(len(metrics))
    width = 0.35
    
    p4_means = []
    p4_stds = []
    prp_means = []
    prp_stds = []
    
    for metric in metrics:
        p4_means.append(stats['single_p4'][metric]['mean'])
        p4_stds.append(stats['single_p4'][metric]['std'])
        prp_means.append(stats['single_prp'][metric]['mean'])
        prp_stds.append(stats['single_prp'][metric]['std'])
    
    # Create bars with consistent colors
    bars1 = ax.bar(x_pos - width/2, p4_means, width,
                   yerr=p4_stds, label='P4',
                   capsize=5, color=COLORS['p4'], alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    bars2 = ax.bar(x_pos + width/2, prp_means, width,
                   yerr=prp_stds, label='Random Pause Player',
                   capsize=5, color=COLORS['random'], alpha=0.8, edgecolor='dimgray', linewidth=1.5)
    
    # Customize plot
    ax.set_xlabel('Score Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score per Turn', fontsize=12, fontweight='bold')
    ax.set_title('P4 vs Random Pause Player: Per-Turn Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Add improvement percentages as text
    improvements = analysis_single['improvements']
    for i, metric in enumerate(metrics):
        if metric in improvements and improvements[metric]['percentage']:
            pct = improvements[metric]['percentage']
            y_pos = max(p4_means[i] + p4_stds[i], prp_means[i] + prp_stds[i]) * 1.05
            ax.text(x_pos[i], y_pos, f'{pct:+.1f}%', 
                   ha='center', fontsize=9, fontweight='bold', color='darkgreen')
    
    # Add value labels
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    
    plt.tight_layout()
    save_path = output_path / 'p4_vs_random_player.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_1v1_matchups(analysis_1v1, output_path):
    """
    Plot 3: 1v1 matchups - side by side comparison for each opponent.
    """
    if analysis_1v1 is None:
        print("⚠ Skipping 1v1 matchups plot - no data available")
        return
    
    by_opponent = analysis_1v1['by_opponent']
    opponents = list(by_opponent.index)
    
    if len(opponents) == 0:
        print("⚠ No 1v1 opponents found")
        return
    
    # Create figure with subplots
    n_opponents = len(opponents)
    fig_height = max(6, 3 * ((n_opponents + 1) // 2))
    
    if n_opponents <= 3:
        fig, axes = plt.subplots(1, n_opponents, figsize=(5*n_opponents, 5))
        if n_opponents == 1:
            axes = [axes]
    else:
        n_cols = 3
        n_rows = (n_opponents + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
        axes = axes.flatten()
    
    for idx, opponent in enumerate(opponents):
        ax = axes[idx]
        
        # Get data for this opponent
        p4_mean = by_opponent.loc[opponent, ('p4_score_total', 'mean')]
        p4_std = by_opponent.loc[opponent, ('p4_score_total', 'std')]
        opp_mean = by_opponent.loc[opponent, ('opponent_score_total', 'mean')]
        opp_std = by_opponent.loc[opponent, ('opponent_score_total', 'std')]
        n_games = int(by_opponent.loc[opponent, ('diff_total', 'count')])
        diff_mean = by_opponent.loc[opponent, ('diff_total', 'mean')]
        
        # Create bars with consistent colors
        x_pos = np.arange(2)
        bars = ax.bar(x_pos, [p4_mean, opp_mean], 
                      yerr=[p4_std, opp_std],
                      capsize=5, alpha=0.8,
                      color=[COLORS['p4'], COLORS['opponent']],
                      edgecolor=['darkgreen', 'darkblue'],
                      linewidth=1.5)
        
        # Customize subplot
        ax.set_title(f'vs {opponent.upper()} (n={n_games})', fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['P4', opponent.upper()])
        ax.set_ylabel('Total Score')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars, [p4_mean, opp_mean]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Add win/loss indicator
        if diff_mean > 0:
            result_text = f'P4 +{diff_mean:.3f}'
            color = 'darkgreen'
        elif diff_mean < 0:
            result_text = f'{opponent.upper()} +{abs(diff_mean):.3f}'
            color = 'darkred'
        else:
            result_text = 'TIE'
            color = 'gray'
        
        ax.text(0.5, 0.95, result_text,
               transform=ax.transAxes,
               ha='center', va='top',
               fontweight='bold', color=color,
               fontsize=10)
    
    # Hide extra subplots if any
    if n_opponents > 1:
        for idx in range(n_opponents, len(axes)):
            axes[idx].set_visible(False)
    
    fig.suptitle('1v1 Head-to-Head Matchups: P4 vs All Opponents', 
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    save_path = output_path / '1v1_matchups_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_1v1_summary(analysis_1v1, output_path):
    """
    Bonus Plot: Overall 1v1 performance summary across all opponents.
    """
    if analysis_1v1 is None:
        return
    
    detailed = analysis_1v1['detailed_results']
    
    if len(detailed) == 0:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Win rate by opponent
    by_opponent = analysis_1v1['by_opponent']
    opponents = list(by_opponent.index)
    win_rates = []
    
    for opp in opponents:
        games = detailed[detailed['opponent'] == opp]
        wins = (games['diff_total'] > 0).sum()
        total = len(games)
        win_rates.append(wins / total * 100 if total > 0 else 0)
    
    # Use gradient of green colors based on win rate
    bar_colors = []
    for rate in win_rates:
        if rate >= 75:
            bar_colors.append('#228B22')  # Forest green for high win rate
        elif rate >= 50:
            bar_colors.append('#90EE90')  # Light green for good win rate
        elif rate >= 25:
            bar_colors.append('#F0E68C')  # Khaki for moderate
        else:
            bar_colors.append('#FFB6C1')  # Light pink for low win rate
    
    bars1 = ax1.bar(range(len(opponents)), win_rates, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_xlabel('Opponent', fontweight='bold')
    ax1.set_ylabel('Win Rate (%)', fontweight='bold')
    ax1.set_title('P4 Win Rate vs Each Opponent', fontweight='bold')
    ax1.set_xticks(range(len(opponents)))
    ax1.set_xticklabels([o.upper() for o in opponents], rotation=45)
    ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% baseline')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Add percentage labels
    for bar, rate in zip(bars1, win_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # Right plot: Score distribution
    ax2.hist(detailed['p4_score_total'], bins=20, alpha=0.7, 
             label='P4 Scores', color=COLORS['p4'], edgecolor='darkgreen', linewidth=1)
    ax2.hist(detailed['opponent_score_total'], bins=20, alpha=0.7,
             label='Opponent Scores', color=COLORS['opponent'], edgecolor='darkblue', linewidth=1)
    ax2.axvline(detailed['p4_score_total'].mean(), color='darkgreen', 
               linestyle='--', linewidth=2, label=f'P4 Mean: {detailed["p4_score_total"].mean():.3f}')
    ax2.axvline(detailed['opponent_score_total'].mean(), color='darkblue',
               linestyle='--', linewidth=2, label=f'Opp Mean: {detailed["opponent_score_total"].mean():.3f}')
    ax2.set_xlabel('Score', fontweight='bold')
    ax2.set_ylabel('Frequency', fontweight='bold')
    ax2.set_title('Score Distribution in 1v1 Games', fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('1v1 Performance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = output_path / '1v1_performance_summary.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_10_unique_vs_9_omit(analysis_group, output_path):
    """
    Plot: 10_unique (with P4) vs 9_omit_p4 (without P4) comparison.
    """
    if analysis_group is None or analysis_group['stats']['10_unique'] is None or analysis_group['stats']['9_omit_p4'] is None:
        print("⚠ Skipping 10_unique vs 9_omit_p4 plot - insufficient data")
        return
    
    stats = analysis_group['stats']
    improvements = analysis_group['improvements']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left plot: Shared scores comparison
    shared_metrics = ['total', 'importance', 'coherence', 'freshness', 'nonmonotonousness']
    shared_labels = ['Total', 'Importance', 'Coherence', 'Freshness', 'Non-mono']
    
    x_pos = np.arange(len(shared_metrics))
    width = 0.35
    
    with_p4_means = []
    with_p4_stds = []
    without_p4_means = []
    without_p4_stds = []
    
    for metric in shared_metrics:
        with_p4_means.append(stats['10_unique']['shared_scores'][metric]['mean'])
        with_p4_stds.append(stats['10_unique']['shared_scores'][metric]['std'])
        without_p4_means.append(stats['9_omit_p4']['shared_scores'][metric]['mean'])
        without_p4_stds.append(stats['9_omit_p4']['shared_scores'][metric]['std'])
    
    bars1 = ax1.bar(x_pos - width/2, with_p4_means, width,
                    yerr=with_p4_stds, label='10 Players (with P4)',
                    capsize=5, color=COLORS['p4'], alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    bars2 = ax1.bar(x_pos + width/2, without_p4_means, width,
                    yerr=without_p4_stds, label='9 Players (without P4)',
                    capsize=5, color=COLORS['without_p4'], alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    ax1.set_xlabel('Shared Score Type', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Score Value', fontsize=11, fontweight='bold')
    ax1.set_title('Shared Scores: Adding P4 as 10th Player', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(shared_labels, rotation=45, ha='right')
    ax1.legend(loc='upper left', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Add improvement percentages
    for i, metric in enumerate(shared_metrics):
        key = f'shared_{metric}'
        if key in improvements and improvements[key]['percentage']:
            pct = improvements[key]['percentage']
            y_pos = max(with_p4_means[i] + with_p4_stds[i], 
                       without_p4_means[i] + without_p4_stds[i]) * 1.05
            color = 'darkgreen' if pct > 0 else 'darkred'
            ax1.text(x_pos[i], y_pos, f'{pct:+.0f}%', 
                    ha='center', fontsize=8, fontweight='bold', color=color)
    
    # Right plot: Player scores comparison
    player_metrics = ['total', 'individual', 'shared']
    player_labels = ['Total', 'Individual', 'Shared']
    
    x_pos2 = np.arange(len(player_metrics))
    
    with_p4_player_means = []
    with_p4_player_stds = []
    without_p4_player_means = []
    without_p4_player_stds = []
    
    for metric in player_metrics:
        with_p4_player_means.append(stats['10_unique']['player_scores'][metric]['mean'])
        with_p4_player_stds.append(stats['10_unique']['player_scores'][metric]['std'])
        without_p4_player_means.append(stats['9_omit_p4']['player_scores'][metric]['mean'])
        without_p4_player_stds.append(stats['9_omit_p4']['player_scores'][metric]['std'])
    
    bars3 = ax2.bar(x_pos2 - width/2, with_p4_player_means, width,
                    yerr=with_p4_player_stds, label='10 Players (with P4)',
                    capsize=5, color=COLORS['p4'], alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
    bars4 = ax2.bar(x_pos2 + width/2, without_p4_player_means, width,
                    yerr=without_p4_player_stds, label='9 Players (without P4)',
                    capsize=5, color=COLORS['without_p4'], alpha=0.8, edgecolor='darkred', linewidth=1.5)
    
    ax2.set_xlabel('Player Score Type', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Average Score per Player', fontsize=11, fontweight='bold')
    ax2.set_title('Average Player Scores: Impact of Adding P4', fontsize=12, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(player_labels)
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Add improvement percentages for player scores
    for i, metric in enumerate(player_metrics):
        key = f'player_{metric}'
        if key in improvements and improvements[key]['percentage']:
            pct = improvements[key]['percentage']
            y_pos = max(with_p4_player_means[i] + with_p4_player_stds[i],
                       without_p4_player_means[i] + without_p4_player_stds[i]) * 1.05
            color = 'darkgreen' if pct > 0 else 'darkred'
            ax2.text(x_pos2[i], y_pos, f'{pct:+.0f}%',
                    ha='center', fontsize=8, fontweight='bold', color=color)
    
    # Add value labels on bars
    def add_value_labels(ax, bars):
        for bar in bars:
            height = bar.get_height()
            if height != 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=7)
    
    add_value_labels(ax1, bars1)
    add_value_labels(ax1, bars2)
    add_value_labels(ax2, bars3)
    add_value_labels(ax2, bars4)
    
    # Add summary text
    n_games_10 = stats['10_unique']['num_games']
    n_games_9 = stats['9_omit_p4']['num_games']
    fig.suptitle(f'Impact of Adding P4: 10 Players (n={n_games_10}) vs 9 Players (n={n_games_9})',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_path = output_path / '10_unique_vs_9_omit_p4.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_all_players_comparison(analysis_all_players, output_path):
    """
    Plot: Comprehensive comparison of all players' normalized per-turn scores.
    """
    if analysis_all_players is None or not analysis_all_players['player_stats']:
        print("⚠ Skipping all players comparison plot - insufficient data")
        return
    
    player_stats = analysis_all_players['player_stats']
    
    # Prepare data - sort players (p1-p10, then prp)
    sorted_players = []
    player_labels = []
    
    for i in range(1, 11):
        if f'p{i}' in player_stats:
            sorted_players.append(f'p{i}')
            player_labels.append(f'P{i}')
    if 'prp' in player_stats:
        sorted_players.append('prp')
        player_labels.append('PRP')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Prepare data for plotting
    x_pos = np.arange(len(sorted_players))
    width = 0.25
    
    # Collect data for all metrics
    total_means = []
    total_stds = []
    shared_means = []
    shared_stds = []
    indiv_means = []
    indiv_stds = []
    game_counts = []
    
    for player in sorted_players:
        stats = player_stats[player]
        total_means.append(stats['player_score_total']['mean'])
        total_stds.append(stats['player_score_total']['std'])
        shared_means.append(stats['shared_score_per_turn']['mean'])
        shared_stds.append(stats['shared_score_per_turn']['std'])
        indiv_means.append(stats['individual_score_per_turn']['mean'])
        indiv_stds.append(stats['individual_score_per_turn']['std'])
        game_counts.append(stats['unique_games'])
    
    # Top plot: All three metrics side by side
    bars1 = ax1.bar(x_pos - width, total_means, width,
                    yerr=total_stds, label='Total (per-turn)',
                    capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x_pos, shared_means, width,
                    yerr=shared_stds, label='Shared (per-turn)',
                    capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    bars3 = ax1.bar(x_pos + width, indiv_means, width,
                    yerr=indiv_stds, label='Individual (per-turn)',
                    capsize=4, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Color bars based on player
    for i, player in enumerate(sorted_players):
        if player == 'p4':
            # P4 gets light green
            bars1[i].set_facecolor(COLORS['p4'])
            bars2[i].set_facecolor(COLORS['p4'])
            bars3[i].set_facecolor(COLORS['p4'])
            bars1[i].set_edgecolor('darkgreen')
            bars2[i].set_edgecolor('darkgreen')
            bars3[i].set_edgecolor('darkgreen')
            bars1[i].set_linewidth(2)
            bars2[i].set_linewidth(2)
            bars3[i].set_linewidth(2)
        elif player == 'prp':
            # PRP gets gray
            bars1[i].set_facecolor(COLORS['random'])
            bars2[i].set_facecolor(COLORS['random'])
            bars3[i].set_facecolor(COLORS['random'])
        else:
            # Other players get sky blue
            bars1[i].set_facecolor(COLORS['opponent'])
            bars2[i].set_facecolor(COLORS['opponent'])
            bars3[i].set_facecolor(COLORS['opponent'])
    
    ax1.set_xlabel('Player', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Score per Turn', fontsize=12, fontweight='bold')
    ax1.set_title('All Players Performance: Normalized Per-Turn Scores', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(player_labels)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add game count annotations
    for i, (player, count) in enumerate(zip(player_labels, game_counts)):
        ax1.text(i, -0.15, f'n={count}', ha='center', va='top', fontsize=7, 
                transform=ax1.get_xaxis_transform())
    
    # Highlight P4 position
    p4_idx = sorted_players.index('p4') if 'p4' in sorted_players else None
    if p4_idx is not None:
        ax1.axvspan(p4_idx - 0.4, p4_idx + 0.4, alpha=0.1, color='green')
    
    # Bottom plot: Total score comparison with ranking
    # Sort by total score for ranking visualization
    player_ranking = [(p, player_stats[p]['player_score_total']['mean'], 
                      player_stats[p]['player_score_total']['std'],
                      player_stats[p]['unique_games']) 
                     for p in sorted_players]
    player_ranking.sort(key=lambda x: x[1], reverse=True)
    
    ranked_labels = []
    ranked_means = []
    ranked_stds = []
    ranked_colors = []
    
    for rank, (player, mean, std, games) in enumerate(player_ranking, 1):
        label = player.upper()
        if player == 'p4':
            label = f'→ {label} ←'
            ranked_colors.append(COLORS['p4'])
        elif player == 'prp':
            ranked_colors.append(COLORS['random'])
        else:
            ranked_colors.append(COLORS['opponent'])
        
        ranked_labels.append(f'{rank}. {label}\n({games} games)')
        ranked_means.append(mean)
        ranked_stds.append(std)
    
    x_pos2 = np.arange(len(ranked_labels))
    bars4 = ax2.bar(x_pos2, ranked_means, yerr=ranked_stds,
                   capsize=5, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Apply colors based on player
    for i, bar in enumerate(bars4):
        bar.set_facecolor(ranked_colors[i])
        if player_ranking[i][0] == 'p4':
            bar.set_edgecolor('darkgreen')
            bar.set_linewidth(2)
    
    ax2.set_xlabel('Player Ranking', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Total Score per Turn', fontsize=12, fontweight='bold')
    ax2.set_title('Player Rankings by Total Score (Sorted)', fontsize=14, fontweight='bold')
    ax2.set_xticks(x_pos2)
    ax2.set_xticklabels(ranked_labels, fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on ranking bars
    for bar, val in zip(bars4, ranked_means):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Add reference lines
    if 'prp' in player_stats:
        prp_mean = player_stats['prp']['player_score_total']['mean']
        ax2.axhline(y=prp_mean, color='gray', linestyle='--', alpha=0.5, 
                   label=f'Random baseline (PRP): {prp_mean:.3f}')
        ax2.legend(loc='upper right')
    
    plt.tight_layout()
    save_path = output_path / 'all_players_comparison.png'
    plt.savefig(save_path, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main(csv_path, output_dir):
    """
    Main function to generate all plots.
    """
    print(f"\n{'='*60}")
    print("GENERATING PLOTS")
    print(f"{'='*60}")
    
    # Import and run the analysis script
    import importlib.util
    import os
    
    # Find the analysis script
    script_dir = Path(__file__).parent
    analysis_script = script_dir / 'simulation_data_loader.py'
    
    if not analysis_script.exists():
        # Try current directory
        analysis_script = Path('simulation_data_loader.py')
        if not analysis_script.exists():
            print("✗ Error: Could not find simulation_data_loader.py")
            print("  Make sure it's in the same directory as this plotting script")
            return
    
    # Load the analysis module
    spec = importlib.util.spec_from_file_location("analysis", analysis_script)
    analysis_module = importlib.util.module_from_spec(spec)
    sys.modules["analysis"] = analysis_module
    spec.loader.exec_module(analysis_module)
    
    # Run the analysis
    print(f"\nRunning analysis on: {csv_path}")
    print("-"*40)
    
    # Temporarily redirect stdout to suppress analysis output
    import io
    from contextlib import redirect_stdout
    
    with redirect_stdout(io.StringIO()):
        # Load and analyze data
        df = analysis_module.load_simulation_data(csv_path)
        df, games_df, summary_stats = analysis_module.prepare_data_for_analysis(df)
        comparison = analysis_module.analyze_score_differences(df, games_df)
        analysis_1v1 = analysis_module.analyze_1v1_with_p4(df)
        analysis_single = analysis_module.analyze_single_p4_vs_prp(df)
        analysis_group = analysis_module.analyze_10_unique_vs_9_omit(df)
        analysis_all_players = analysis_module.analyze_all_players_performance(df)
    
    print("✓ Analysis complete")
    
    # Create output directory
    output_path = ensure_output_dir(output_dir)
    print(f"\nSaving plots to: {output_path}")
    print("-"*40)
    
    # Generate plots
    plot_scores_with_without_p4(comparison, output_path)
    plot_p4_vs_random(analysis_single, output_path)
    plot_1v1_matchups(analysis_1v1, output_path)
    plot_1v1_summary(analysis_1v1, output_path)
    plot_10_unique_vs_9_omit(analysis_group, output_path)
    plot_all_players_comparison(analysis_all_players, output_path)
    
    print(f"\n{'='*60}")
    print(f"✓ All plots saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python plot_results.py <path_to_csv> <output_directory>")
        sys.exit(1)
    
    csv_path = Path(sys.argv[1])
    output_dir = sys.argv[2]
    
    if not csv_path.exists():
        print(f"Error: CSV file '{csv_path}' not found")
        sys.exit(1)
    
    main(csv_path, output_dir)