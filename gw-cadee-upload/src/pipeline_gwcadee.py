"""
Pipeline 2: GW-CADEE (Distance Correlation-Based)
Modified algorithm with only 2 changes from CADEE.

This pipeline can process any dataset with the following structure:
- Input: CSV file or pandas DataFrame with feature pairs
- Output: Mutual information estimates for each pair

Usage:
    from pipeline_gwcadee import GWCADEEPipeline
    
    pipeline = GWCADEEPipeline()
    results = pipeline.run_dataset('my_data.csv')
    results.to_csv('gwcadee_results.csv')
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class GWCADEEPipeline:
    """
    GW-CADEE Pipeline: Distance correlation-based mutual information estimator.
    
    Key characteristics:
    - Independence test: dCor < adaptive_threshold(n)  ← CHANGED from CADEE
    - Split selection: max dCor  ← CHANGED from CADEE
    - Copula transformation: rank-based (SAME as CADEE)
    - Recursive partitioning: binary splits (SAME as CADEE)
    
    Only 2 steps differ from CADEE: Steps 4 and 6!
    """
    
    def __init__(self, max_depth=4, min_samples=30, use_adaptive=True):
        """
        Initialize GW-CADEE pipeline.
        
        Parameters:
        -----------
        max_depth : int
            Maximum recursion depth (default: 4)
        min_samples : int
            Minimum samples per partition (default: 30)
        use_adaptive : bool
            Use adaptive thresholds based on sample size (default: True)
        """
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.use_adaptive = use_adaptive
        self.name = "GW-CADEE"
        
    def copula_transform(self, x):
        """Transform to copula space (rank-based). SAME AS CADEE."""
        n = len(x)
        ranks = np.argsort(np.argsort(x))
        return (ranks + 1) / (n + 1)
    
    def spacing_entropy(self, x):
        """Estimate entropy using spacing estimator. SAME AS CADEE."""
        x_sorted = np.sort(x)
        n = len(x_sorted)
        if n < 2:
            return 0.0
        spacings = np.diff(x_sorted)
        spacings = spacings[spacings > 0]
        if len(spacings) == 0:
            return 0.0
        return np.log(n) + np.mean(np.log(spacings + 1e-10))
    
    def distance_correlation(self, x, y):
        """
        Compute distance correlation (Székely et al., 2007).
        
        This is the KEY DIFFERENCE from CADEE (Step 4 & 6).
        
        Parameters:
        -----------
        x, y : array-like
            Input vectors
            
        Returns:
        --------
        dcor : float
            Distance correlation [0, 1]
        """
        n = len(x)
        
        # Compute pairwise Euclidean distances
        a = squareform(pdist(x.reshape(-1, 1), metric='euclidean'))
        b = squareform(pdist(y.reshape(-1, 1), metric='euclidean'))
        
        # Double-center (U-centering)
        A = a - a.mean(axis=0, keepdims=True) - a.mean(axis=1, keepdims=True) + a.mean()
        B = b - b.mean(axis=0, keepdims=True) - b.mean(axis=1, keepdims=True) + b.mean()
        
        # Distance covariance squared
        dcov2_xy = (A * B).sum() / (n * n)
        dcov2_xx = (A * A).sum() / (n * n)
        dcov2_yy = (B * B).sum() / (n * n)
        
        # Distance correlation
        if dcov2_xx * dcov2_yy > 0:
            return np.sqrt(dcov2_xy) / np.sqrt(np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy))
        return 0.0
    
    def adaptive_threshold(self, n):
        """
        Adaptive threshold based on sample size.
        
        This enables GW-CADEE to work on smaller samples than CADEE.
        
        Parameters:
        -----------
        n : int
            Sample size
            
        Returns:
        --------
        threshold : float
            Adaptive threshold for independence test
        """
        if not self.use_adaptive:
            return 0.05
        
        if n >= 1000:
            return 0.05
        elif n >= 500:
            return 0.03
        elif n >= 200:
            return 0.02
        elif n >= 100:
            return 0.01
        else:
            return 0.005
    
    def gwcadee_recursive(self, u, v, depth=0):
        """
        GW-CADEE recursive algorithm.
        
        ONLY Steps 4 and 6 differ from CADEE!
        
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
        
        # Step 3: Stopping criteria (SAME AS CADEE)
        if n < self.min_samples or depth >= self.max_depth:
            return 0.0
        
        # Step 4: Independence test (DISTANCE CORRELATION - GW-CADEE specific)
        dcor = self.distance_correlation(u, v)
        threshold = self.adaptive_threshold(n)
        
        if dcor < threshold:
            return 0.0
        
        # Step 5: Entropy computation (SAME AS CADEE)
        h_u = self.spacing_entropy(u)
        h_v = self.spacing_entropy(v)
        
        # Step 6: Split selection (DISTANCE CORRELATION - GW-CADEE specific)
        dcor_u = self.distance_correlation(u, v)
        dcor_v = self.distance_correlation(v, u)
        
        if dcor_u > dcor_v:
            split_var = u
        else:
            split_var = v
        
        # Step 7: Binary partition (SAME AS CADEE)
        median = np.median(split_var)
        left_idx = split_var <= median
        
        if left_idx.sum() == 0 or (~left_idx).sum() == 0:
            return 0.0
        
        # Step 8: Rescale (SAME AS CADEE)
        u_left = 2 * u[left_idx]
        v_left = 2 * v[left_idx]
        u_right = 2 * u[~left_idx] - 1
        v_right = 2 * v[~left_idx] - 1
        
        # Step 9: Recursive calls (SAME AS CADEE)
        mi_left = self.gwcadee_recursive(u_left, v_left, depth + 1)
        mi_right = self.gwcadee_recursive(u_right, v_right, depth + 1)
        
        # Step 10: Aggregate (SAME AS CADEE)
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
        # Step 1: Copula transform (SAME AS CADEE)
        u = self.copula_transform(x)
        v = self.copula_transform(y)
        
        # Step 2: Initialize and run recursion
        mi = self.gwcadee_recursive(u, v, depth=0)
        
        return mi
    
    def run_dataset(self, data, pairs=None, verbose=True):
        """
        Run GW-CADEE on a dataset.
        
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
        Run GW-CADEE on Ariel's synthetic dataset.
        
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
        """Generate Ariel's 11 synthetic patterns. SAME AS CADEE."""
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
    print("GW-CADEE PIPELINE EXAMPLE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = GWCADEEPipeline()
    
    # Example 1: Run on Ariel's synthetic data
    print("\nExample 1: Ariel's Synthetic Dataset")
    print("-"*80)
    ariel_results = pipeline.run_ariel_synthetic(n=1000)
    print(f"\nDetection rate: {ariel_results['Detected'].sum()}/{len(ariel_results)}")
    
    # Example 2: Run on custom data
    print("\n" + "="*80)
    print("Example 2: Custom Dataset")
    print("-"*80)
    
    # Create sample data with non-monotonic relationship
    sample_data = pd.DataFrame({
        'feature1': np.random.uniform(-2, 2, 500),
    })
    sample_data['feature2'] = sample_data['feature1']**2 + np.random.randn(500) * 0.3  # Quadratic
    sample_data['feature3'] = np.random.randn(500)  # Independent
    sample_data['feature4'] = sample_data['feature1'] * 2 + np.random.randn(500) * 0.1  # Linear
    
    custom_results = pipeline.run_dataset(sample_data)
    print("\nResults:")
    print(custom_results)
    
    # Save results
    ariel_results.to_csv('/mnt/user-data/outputs/pipeline_gwcadee_ariel_results.csv', index=False)
    custom_results.to_csv('/mnt/user-data/outputs/pipeline_gwcadee_custom_results.csv', index=False)
    
    print("\n" + "="*80)
    print("Results saved!")
    print("="*80)
