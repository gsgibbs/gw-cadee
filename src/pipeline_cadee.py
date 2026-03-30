"""
Pipeline 1: CADEE (Spearman Correlation-Based)
Original algorithm from the paper, converted to Python.

This pipeline can process any dataset with the following structure:
- Input: CSV file or pandas DataFrame with feature pairs
- Output: Mutual information estimates for each pair

Usage:
    from pipeline_cadee import CADEEPipeline
    
    pipeline = CADEEPipeline()
    results = pipeline.run_dataset('my_data.csv')
    results.to_csv('cadee_results.csv')
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

class CADEEPipeline:
    """
    CADEE Pipeline: Spearman correlation-based mutual information estimator.
    
    Key characteristics:
    - Independence test: Spearman p-value > 0.05
    - Split selection: max |Spearman ρ|
    - Copula transformation: rank-based
    - Recursive partitioning: binary splits
    """
    
    def __init__(self, max_depth=4, min_samples=30, threshold=0.05):
        """
        Initialize CADEE pipeline.
        
        Parameters:
        -----------
        max_depth : int
            Maximum recursion depth (default: 4)
        min_samples : int
            Minimum samples per partition (default: 30)
        threshold : float
            Spearman p-value threshold for independence (default: 0.05)
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.threshold = threshold
        self.name = "CADEE"
        
    def copula_transform(self, x):
        """Transform to copula space (rank-based)."""
        n = len(x)
        ranks = np.argsort(np.argsort(x))
        return (ranks + 1) / (n + 1)
    
    def spacing_entropy(self, x):
        """Estimate entropy using spacing estimator."""
        x_sorted = np.sort(x)
        n = len(x_sorted)
        if n < 2:
            return 0.0
        spacings = np.diff(x_sorted)
        spacings = spacings[spacings > 0]
        if len(spacings) == 0:
            return 0.0
        return np.log(n) + np.mean(np.log(spacings + 1e-10))
    
    def cadee_recursive(self, u, v, depth=0):
        """
        CADEE recursive algorithm.
        
        Parameters:
        -----------
        u, v : array-like
            Copula-transformed data in [0,1]
        depth : int
            Current recursion depth
            
        Returns:
        --------
        mi : float
            Mutual information estimate
        """
        n = len(u)
        
        # Step 3: Stopping criteria
        if n < self.min_samples or depth >= self.max_depth:
            return 0.0
        
        # Step 4: Independence test (SPEARMAN - CADEE specific)
        rho, p_value = spearmanr(u, v)
        if p_value > self.threshold:
            return 0.0
        
        # Step 5: Entropy computation
        h_u = self.spacing_entropy(u)
        h_v = self.spacing_entropy(v)
        
        # Step 6: Split selection (SPEARMAN - CADEE specific)
        rho_u = abs(spearmanr(u, v)[0])
        rho_v = abs(spearmanr(v, u)[0])
        
        if rho_u > rho_v:
            split_var = u
        else:
            split_var = v
        
        # Step 7: Binary partition
        median = np.median(split_var)
        left_idx = split_var <= median
        
        if left_idx.sum() == 0 or (~left_idx).sum() == 0:
            return 0.0
        
        # Step 8: Rescale
        u_left = 2 * u[left_idx]
        v_left = 2 * v[left_idx]
        u_right = 2 * u[~left_idx] - 1
        v_right = 2 * v[~left_idx] - 1
        
        # Step 9: Recursive calls
        mi_left = self.cadee_recursive(u_left, v_left, depth + 1)
        mi_right = self.cadee_recursive(u_right, v_right, depth + 1)
        
        # Step 10: Aggregate
        p_left = left_idx.sum() / n
        mi = p_left * mi_left + (1 - p_left) * mi_right + h_u + h_v
        
        return max(0, mi)
    
    def compute_mi(self, x, y):
        """
        Compute mutual information for a pair of variables.
        
        Parameters:
        -----------
        x, y : array-like
            Raw data vectors
            
        Returns:
        --------
        mi : float
            Mutual information estimate
        """
        # Step 1: Copula transform
        u = self.copula_transform(x)
        v = self.copula_transform(y)
        
        # Step 2: Initialize and run recursion
        mi = self.cadee_recursive(u, v, depth=0)
        
        return mi
    
    def run_dataset(self, data, pairs=None, verbose=True):
        """
        Run CADEE on a dataset.
        
        Parameters:
        -----------
        data : str or DataFrame
            Path to CSV file or pandas DataFrame
        pairs : list of tuples, optional
            List of (col1, col2) pairs to analyze.
            If None, analyzes all pairwise combinations.
        verbose : bool
            Print progress messages
            
        Returns:
        --------
        results : DataFrame
            Results with columns: Feature1, Feature2, MI, Method
        """
        # Load data
        if isinstance(data, str):
            df = pd.read_csv(data)
        else:
            df = data.copy()
        
        if verbose:
            print(f"Running {self.name} pipeline on dataset with {len(df)} samples")
        
        # Generate pairs if not provided
        if pairs is None:
            columns = df.columns
            pairs = [(columns[i], columns[j]) 
                    for i in range(len(columns)) 
                    for j in range(i+1, len(columns))]
        
        results = []
        
        for idx, (col1, col2) in enumerate(pairs):
            if verbose and (idx + 1) % 10 == 0:
                print(f"  Processing pair {idx + 1}/{len(pairs)}: {col1} × {col2}")
            
            x = df[col1].values
            y = df[col2].values
            
            # Remove NaN
            mask = ~(np.isnan(x) | np.isnan(y))
            x = x[mask]
            y = y[mask]
            
            if len(x) < self.min_samples:
                mi = 0.0
            else:
                mi = self.compute_mi(x, y)
            
            results.append({
                'Feature1': col1,
                'Feature2': col2,
                'MI': mi,
                'Method': self.name,
                'Detected': mi > 0.05
            })
        
        results_df = pd.DataFrame(results)
        
        if verbose:
            detected = results_df['Detected'].sum()
            print(f"\n{self.name} Results:")
            print(f"  Total pairs: {len(results_df)}")
            print(f"  Detected: {detected}/{len(results_df)} ({detected/len(results_df)*100:.1f}%)")
            print(f"  Average MI: {results_df['MI'].mean():.3f}")
        
        return results_df
    
    def run_ariel_synthetic(self, n=1000, verbose=True):
        """
        Run CADEE on Ariel's synthetic dataset.
        
        Parameters:
        -----------
        n : int
            Number of samples
        verbose : bool
            Print progress
            
        Returns:
        --------
        results : DataFrame
            Results for all 11 Ariel patterns
        """
        if verbose:
            print("Generating Ariel's synthetic dataset...")
        
        patterns = self._generate_ariel_data(n)
        
        results = []
        for name, x, y, relationship_type in patterns:
            mi = self.compute_mi(x, y)
            results.append({
                'Relationship': name,
                'Type': relationship_type,
                'MI': mi,
                'Method': self.name,
                'Detected': mi > 0.05
            })
            
            if verbose:
                print(f"  {name:20s} | Type: {relationship_type:15s} | MI: {mi:.3f}")
        
        return pd.DataFrame(results)
    
    def _generate_ariel_data(self, n):
        """Generate Ariel's 11 synthetic patterns."""
        np.random.seed(42)
        results = []
        
        # 1. Linear positive
        x = np.random.randn(n)
        y = 2*x + np.random.randn(n)*0.5
        results.append(('Linear Positive', x, y, 'Monotonic'))
        
        # 2. Linear negative
        x = np.random.randn(n)
        y = -2*x + np.random.randn(n)*0.5
        results.append(('Linear Negative', x, y, 'Monotonic'))
        
        # 3. Quadratic
        x = np.random.uniform(-2, 2, n)
        y = x**2 + np.random.randn(n)*0.3
        results.append(('Quadratic', x, y, 'Non-monotonic'))
        
        # 4. Cubic
        x = np.random.uniform(-2, 2, n)
        y = x**3 + np.random.randn(n)*0.5
        results.append(('Cubic', x, y, 'Non-monotonic'))
        
        # 5. Sine wave
        x = np.random.uniform(-np.pi, np.pi, n)
        y = np.sin(x) + np.random.randn(n)*0.2
        results.append(('Sine Wave', x, y, 'Non-monotonic'))
        
        # 6. Circle
        theta = np.random.uniform(0, 2*np.pi, n)
        r = 1 + np.random.randn(n)*0.1
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        results.append(('Circle', x, y, 'Non-monotonic'))
        
        # 7. X-shape
        x = np.random.uniform(-2, 2, n)
        y = (x**2 + np.random.uniform(0, 0.3, n)) * np.random.choice([-1, 1], n)
        results.append(('X-Shape', x, y, 'Non-monotonic'))
        
        # 8. Step function
        x = np.random.uniform(-2, 2, n)
        y = np.sign(x) + np.random.randn(n)*0.2
        results.append(('Step Function', x, y, 'Non-monotonic'))
        
        # 9. Exponential
        x = np.random.uniform(0, 3, n)
        y = np.exp(x) + np.random.randn(n)*0.5
        results.append(('Exponential', x, y, 'Monotonic'))
        
        # 10. Logarithmic
        x = np.random.uniform(0.1, 10, n)
        y = np.log(x) + np.random.randn(n)*0.2
        results.append(('Logarithmic', x, y, 'Monotonic'))
        
        # 11. Independent
        x = np.random.randn(n)
        y = np.random.randn(n)
        results.append(('Independent', x, y, 'Independent'))
        
        return results


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("="*80)
    print("CADEE PIPELINE EXAMPLE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = CADEEPipeline()
    
    # Example 1: Run on Ariel's synthetic data
    print("\nExample 1: Ariel's Synthetic Dataset")
    print("-"*80)
    ariel_results = pipeline.run_ariel_synthetic(n=1000)
    print(f"\nDetection rate: {ariel_results['Detected'].sum()}/{len(ariel_results)}")
    
    # Example 2: Run on custom data
    print("\n" + "="*80)
    print("Example 2: Custom Dataset")
    print("-"*80)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.randn(500),
        'feature2': np.random.randn(500),
        'feature3': np.random.uniform(0, 10, 500)
    })
    sample_data['feature4'] = sample_data['feature1'] * 2 + np.random.randn(500) * 0.1
    
    custom_results = pipeline.run_dataset(sample_data)
    print("\nResults:")
    print(custom_results)
    
    # Save results
    ariel_results.to_csv('/mnt/user-data/outputs/pipeline_cadee_ariel_results.csv', index=False)
    custom_results.to_csv('/mnt/user-data/outputs/pipeline_cadee_custom_results.csv', index=False)
    
    print("\n" + "="*80)
    print("Results saved!")
    print("="*80)
