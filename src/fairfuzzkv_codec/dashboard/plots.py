import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

from fairfuzzkv_codec.repair_scoring.fuzzy import INPUT_LEVELS, OUTPUT_LEVELS, triangular

def plot_distortion_vs_budget(results: List[Dict], output_dir: str):
    """
    Plots numerical KV distortion vs bit-budget for different baselines.
    results: list of dicts with keys: 'codec_name', 'bits_per_token', 'distortion'
    """
    plt.figure(figsize=(10, 6))
    
    # Group by codec
    codecs = set([r['codec_name'] for r in results])
    
    for codec in codecs:
        codec_results = [r for r in results if r['codec_name'] == codec]
        # Sort by bits_per_token
        codec_results.sort(key=lambda x: x['bits_per_token'])
        
        x = [r['bits_per_token'] for r in codec_results]
        y = [r['distortion'] for r in codec_results]
        
        plt.plot(x, y, marker='o', label=codec)
        
    plt.xlabel('Bits per Element')
    plt.ylabel('KV Distortion (MSE)')
    plt.title('KV Distortion vs Bit Budget')
    plt.legend()
    plt.grid(True)
    
    out_path = os.path.join(output_dir, 'distortion_vs_budget.png')
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_compression_ratio(results: List[Dict], output_dir: str):
    """
    Plots byte-level compression ratios.
    results: list of dicts with keys: 'codec_name', 'compression_ratio'
    """
    plt.figure(figsize=(8, 6))
    
    codecs = [r['codec_name'] for r in results]
    ratios = [r['compression_ratio'] for r in results]
    
    plt.bar(codecs, ratios, color='skyblue')
    plt.xlabel('Codec Baseline')
    plt.ylabel('Compression Ratio (Original Size / Serialized Size)')
    plt.title('Byte-level Compression Ratio')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    out_path = os.path.join(output_dir, 'compression_ratio.png')
    plt.savefig(out_path)
    plt.close()
    return out_path

def plot_rate_distortion_curves(results: List[Dict], output_dir: str, metric_key: str = 'mse', ylabel: str = 'MSE'):
    """
    Rate-distortion curve: bits/element vs a distortion metric, one line per
    tensor ('K', 'V', or 'task') so K/V/task-level curves can be compared
    on one plot or called separately per tensor.
    results: list of dicts with keys: 'series_name', 'bits_per_element', metric_key
    """
    plt.figure(figsize=(10, 6))

    series_names = sorted({r['series_name'] for r in results})
    for series in series_names:
        series_results = [r for r in results if r['series_name'] == series]
        series_results.sort(key=lambda x: x['bits_per_element'])
        x = [r['bits_per_element'] for r in series_results]
        y = [r[metric_key] for r in series_results]
        plt.plot(x, y, marker='o', label=series)

    plt.xlabel('Bits per Element')
    plt.ylabel(ylabel)
    plt.title(f'Rate-Distortion: {ylabel} vs Bit Budget')
    plt.legend()
    plt.grid(True)

    out_path = os.path.join(output_dir, f'rate_distortion_{metric_key}.png')
    plt.savefig(out_path)
    plt.close()
    return out_path


def plot_fuzzy_membership_functions(
    output_dir: str,
    input_levels: Dict[str, Tuple[float, float, float]] = INPUT_LEVELS,
    output_levels: Dict[str, Tuple[float, float, float]] = OUTPUT_LEVELS,
    filename: str = 'fuzzy_membership.png',
) -> str:
    """Plots the Module 3 fuzzy scorer's input (low/high) and output
    (low/medium/high priority) membership functions side by side, so the
    rule base is visually inspectable, not just a black box."""
    xs = [i / 200 for i in range(201)]
    fig, (ax_in, ax_out) = plt.subplots(1, 2, figsize=(12, 5))

    for name, bp in input_levels.items():
        ax_in.plot(xs, [triangular(x, *bp) for x in xs], label=name)
    ax_in.set_xlabel('Normalized Input Value')
    ax_in.set_ylabel('Membership Degree')
    ax_in.set_title('Input Membership Functions')
    ax_in.legend()
    ax_in.grid(True)

    for name, bp in output_levels.items():
        ax_out.plot(xs, [triangular(x, *bp) for x in xs], label=name)
    ax_out.set_xlabel('Repair Priority')
    ax_out.set_ylabel('Membership Degree')
    ax_out.set_title('Output Membership Functions')
    ax_out.legend()
    ax_out.grid(True)

    fig.tight_layout()
    out_path = os.path.join(output_dir, filename)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
