"""
Master Comparison Script
Runs all three pipelines (CADEE, GW-CADEE, NMI) on any dataset and compares results.

Usage:
    python run_all_pipelines.py --data my_data.csv --output results/
    
Or import and use programmatically:
    from run_all_pipelines import run_all_pipelines
    results = run_all_pipelines('my_data.csv')
"""

import numpy as np
import pandas as pd
import sys
import os

# Import all three pipelines
sys.path.insert(0, '/home/claude')
from pipeline_cadee import CADEEPipeline
from pipeline_gwcadee import GWCADEEPipeline
from pipeline_nmi import NMIPipeline

def run_all_pipelines(data, pairs=None, output_dir='/mnt/user-data/outputs/', 
                      save_individual=True, verbose=True):
    """
    Run all three pipelines on a dataset and compare results.
    
    Parameters:
    -----------
    data : str or DataFrame
        Path to CSV file or pandas DataFrame
    pairs : list of tuples, optional
        Specific pairs to analyze. If None, analyzes all combinations.
    output_dir : str
        Directory to save results
    save_individual : bool
        Save individual pipeline results
    verbose : bool
        Print progress
        
    Returns:
    --------
    comparison : DataFrame
        Combined results from all three methods
    """
    if verbose:
        print("="*80)
        print("RUNNING ALL THREE PIPELINES")
        print("="*80)
    
    # Initialize pipelines
    cadee = CADEEPipeline()
    gwcadee = GWCADEEPipeline()
    nmi = NMIPipeline()
    
    # Run each pipeline
    if verbose:
        print("\n[1/3] Running CADEE...")
    cadee_results = cadee.run_dataset(data, pairs=pairs, verbose=verbose)
    
    if verbose:
        print("\n[2/3] Running GW-CADEE...")
    gwcadee_results = gwcadee.run_dataset(data, pairs=pairs, verbose=verbose)
    
    if verbose:
        print("\n[3/3] Running Normalized MI...")
    nmi_results = nmi.run_dataset(data, pairs=pairs, method='discretized', verbose=verbose)
    
    # Combine results
    comparison = pd.DataFrame({
        'Feature1': cadee_results['Feature1'],
        'Feature2': cadee_results['Feature2'],
        'CADEE': cadee_results['MI'],
        'GW-CADEE': gwcadee_results['MI'],
        'NMI': nmi_results['MI'],
        'CADEE_Detected': cadee_results['Detected'],
        'GWCADEE_Detected': gwcadee_results['Detected'],
        'NMI_Detected': nmi_results['Detected']
    })
    
    # Add improvement metrics
    comparison['Improvement_Abs'] = comparison['GW-CADEE'] - comparison['CADEE']
    comparison['Improvement_Pct'] = (
        (comparison['GW-CADEE'] - comparison['CADEE']) / 
        (comparison['CADEE'] + 1e-10) * 100
    )
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    if save_individual:
        cadee_results.to_csv(f'{output_dir}/cadee_results.csv', index=False)
        gwcadee_results.to_csv(f'{output_dir}/gwcadee_results.csv', index=False)
        nmi_results.to_csv(f'{output_dir}/nmi_results.csv', index=False)
    
    comparison.to_csv(f'{output_dir}/comparison_all_methods.csv', index=False)
    
    if verbose:
        print("\n" + "="*80)
        print("COMPARISON SUMMARY")
        print("="*80)
        
        print(f"\nTotal pairs analyzed: {len(comparison)}")
        print(f"\nDetection rates:")
        print(f"  CADEE:    {comparison['CADEE_Detected'].sum()}/{len(comparison)} ({comparison['CADEE_Detected'].mean()*100:.1f}%)")
        print(f"  GW-CADEE: {comparison['GWCADEE_Detected'].sum()}/{len(comparison)} ({comparison['GWCADEE_Detected'].mean()*100:.1f}%)")
        print(f"  NMI:      {comparison['NMI_Detected'].sum()}/{len(comparison)} ({comparison['NMI_Detected'].mean()*100:.1f}%)")
        
        print(f"\nAverage MI values:")
        print(f"  CADEE:    {comparison['CADEE'].mean():.3f}")
        print(f"  GW-CADEE: {comparison['GW-CADEE'].mean():.3f}")
        print(f"  NMI:      {comparison['NMI'].mean():.3f}")
        
        improvement = comparison['GWCADEE_Detected'].sum() - comparison['CADEE_Detected'].sum()
        print(f"\nGW-CADEE detected {improvement} additional dependencies over CADEE")
        
        print(f"\nResults saved to: {output_dir}")
    
    return comparison

def run_ariel_comparison(n=1000, output_dir='/mnt/user-data/outputs/', verbose=True):
    """
    Run all three pipelines on Ariel's synthetic dataset.
    
    Parameters:
    -----------
    n : int
        Number of samples
    output_dir : str
        Directory to save results
    verbose : bool
        Print progress
        
    Returns:
    --------
    comparison : DataFrame
        Combined results from all three methods
    """
    if verbose:
        print("="*80)
        print("ARIEL'S SYNTHETIC DATASET COMPARISON")
        print("="*80)
    
    # Initialize pipelines
    cadee = CADEEPipeline()
    gwcadee = GWCADEEPipeline()
    nmi = NMIPipeline()
    
    # Run each pipeline
    if verbose:
        print("\n[1/3] CADEE Results:")
        print("-"*80)
    cadee_results = cadee.run_ariel_synthetic(n=n, verbose=verbose)
    
    if verbose:
        print("\n[2/3] GW-CADEE Results:")
        print("-"*80)
    gwcadee_results = gwcadee.run_ariel_synthetic(n=n, verbose=verbose)
    
    if verbose:
        print("\n[3/3] Normalized MI Results:")
        print("-"*80)
    nmi_results = nmi.run_ariel_synthetic(n=n, method='discretized', verbose=verbose)
    
    # Combine results
    comparison = pd.DataFrame({
        'Relationship': cadee_results['Relationship'],
        'Type': cadee_results['Type'],
        'CADEE': cadee_results['MI'],
        'GW-CADEE': gwcadee_results['MI'],
        'NMI': nmi_results['MI'],
        'CADEE_Detected': cadee_results['Detected'],
        'GWCADEE_Detected': gwcadee_results['Detected'],
        'NMI_Detected': nmi_results['Detected']
    })
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    comparison.to_csv(f'{output_dir}/ariel_comparison_all_methods.csv', index=False)
    
    if verbose:
        print("\n" + "="*80)
        print("ARIEL DATASET SUMMARY")
        print("="*80)
        
        # Overall
        print(f"\nOverall Detection Rates:")
        print(f"  CADEE:    {comparison['CADEE_Detected'].sum()}/11 ({comparison['CADEE_Detected'].mean()*100:.1f}%)")
        print(f"  GW-CADEE: {comparison['GWCADEE_Detected'].sum()}/11 ({comparison['GWCADEE_Detected'].mean()*100:.1f}%)")
        print(f"  NMI:      {comparison['NMI_Detected'].sum()}/11 ({comparison['NMI_Detected'].mean()*100:.1f}%)")
        
        # By type
        for relationship_type in ['Monotonic', 'Non-monotonic', 'Independent']:
            subset = comparison[comparison['Type'] == relationship_type]
            if len(subset) > 0:
                print(f"\n{relationship_type}:")
                print(f"  CADEE:    {subset['CADEE_Detected'].sum()}/{len(subset)}")
                print(f"  GW-CADEE: {subset['GWCADEE_Detected'].sum()}/{len(subset)}")
                print(f"  NMI:      {subset['NMI_Detected'].sum()}/{len(subset)}")
        
        print(f"\nResults saved to: {output_dir}ariel_comparison_all_methods.csv")
    
    return comparison


# ==================== COMMAND LINE INTERFACE ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run all three MI estimation pipelines')
    parser.add_argument('--data', type=str, help='Path to dataset CSV file')
    parser.add_argument('--ariel', action='store_true', help='Run on Ariel synthetic dataset')
    parser.add_argument('--output', type=str, default='/mnt/user-data/outputs/', 
                       help='Output directory')
    parser.add_argument('--n', type=int, default=1000, 
                       help='Number of samples for Ariel dataset')
    
    args = parser.parse_args()
    
    if args.ariel:
        # Run Ariel comparison
        comparison = run_ariel_comparison(n=args.n, output_dir=args.output)
        
    elif args.data:
        # Run on custom dataset
        comparison = run_all_pipelines(args.data, output_dir=args.output)
        
    else:
        # Demo mode
        print("="*80)
        print("DEMO MODE: Running on sample dataset")
        print("="*80)
        print("\nUsage:")
        print("  python run_all_pipelines.py --ariel")
        print("  python run_all_pipelines.py --data my_data.csv")
        print("\n" + "="*80)
        
        # Run Ariel comparison as demo
        comparison = run_ariel_comparison(n=1000)
