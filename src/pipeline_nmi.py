"""
Pipeline 3: Normalized MI (KSG Estimator)
k-Nearest Neighbors approach, completely different from CADEE/GW-CADEE.

This pipeline can process any dataset with the following structure:
- Input: CSV file or pandas DataFrame with feature pairs
- Output: Normalized mutual information estimates for each pair

Usage:
    from pipeline_nmi import NMIPipeline
    
    pipeline = NMIPipeline()
    results = pipeline.run_dataset('my_data.csv')
    results.to_csv('nmi_results.csv')
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import normalized_mutual_info_score
import warnings
warnings.filterwarnings('ignore')

class NMIPipeline:
    """
    Normalized MI Pipeline: k-Nearest Neighbors estimator (KSG).
    
    Key characteristics:
    - No copula transformation (uses raw data)
    - No recursive partitioning
    - Direct k-NN estimation
    - Output: [0, 1] normalized
    
    Completely different approach from CADEE/GW-CADEE!
    """
    
    def __init__(self, k=3, normalize=True):
        """
        Initialize NMI pipeline.
        
        Parameters:
        -----------
        k : int
            Number of nearest neighbors (default: 3)
        normalize : bool
            Return normalized MI in [0,1] (default: True)
        """
        self.k = k
        self.normalize = normalize
        self.name = "NMI"
        
    def discretize(self, x, bins=10):
        """
        Discretize continuous data into bins.
        
        Parameters:
        -----------
        x : array-like
            Continuous data
        bins : int
            Number of bins
            
        Returns:
        --------
        x_discrete : array
            Discretized data
        """
        return np.digitize(x, np.linspace(x.min(), x.max(), bins))
    
    def entropy(self, x, bins=30):
        """
        Estimate entropy using histogram.
        
        Parameters:
        -----------
        x : array-like
            Input data
        bins : int
            Number of bins
            
        Returns:
        --------
        h : float
            Entropy estimate
        """
        hist, _ = np.histogram(x, bins=bins)
        hist = hist[hist > 0]
        probs = hist / hist.sum()
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    def compute_mi_knn(self, x, y):
        """
        Compute mutual information using k-NN (sklearn).
        
        This is the KSG estimator approach.
        
        Parameters:
        -----------
        x, y : array-like
            Raw data vectors
            
        Returns:
        --------
        mi : float
            Mutual information estimate
        """
        # Use sklearn's mutual_info_regression
        # This implements the KSG estimator
        mi = mutual_info_regression(
            x.reshape(-1, 1), 
            y, 
            n_neighbors=self.k,
            random_state=42
        )[0]
        
        return mi
    
    def compute_nmi_discretized(self, x, y, bins=10):
        """
        Compute normalized MI using discretization.
        
        Alternative method using binning instead of k-NN.
        
        Parameters:
        -----------
        x, y : array-like
            Raw data vectors
        bins : int
            Number of bins for discretization
            
        Returns:
        --------
        nmi : float
            Normalized mutual information [0, 1]
        """
        # Discretize
        x_discrete = self.discretize(x, bins=bins)
        y_discrete = self.discretize(y, bins=bins)
        
        # Compute NMI using sklearn
        nmi = normalized_mutual_info_score(
            x_discrete, 
            y_discrete, 
            average_method='arithmetic'
        )
        
        return nmi
    
    def compute_mi(self, x, y, method='knn'):
        """
        Compute mutual information for a pair of variables.
        
        Parameters:
        -----------
        x, y : array-like
            Raw data vectors
        method : str
            'knn' for k-NN estimator (KSG)
            'discretized' for histogram-based
            
        Returns:
        --------
        mi : float
            Mutual information estimate
        """
        if method == 'knn':
            # k-NN approach (KSG estimator)
            mi = self.compute_mi_knn(x, y)
            
            if self.normalize:
                # Normalize by marginal entropies
                h_x = self.entropy(x)
                h_y = self.entropy(y)
                
                if min(h_x, h_y) > 0:
                    mi = mi / min(h_x, h_y)
                    mi = np.clip(mi, 0, 1)  # Bound to [0,1]
        
        elif method == 'discretized':
            # Discretization approach
            mi = self.compute_nmi_discretized(x, y)
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        return mi
    
    def run_dataset(self, data, pairs=None, method='discretized', verbose=True):
        """
        Run NMI on a dataset.
        
        Parameters:
        -----------
        data : str or DataFrame
            Path to CSV file or pandas DataFrame
        pairs : list of tuples, optional
            List of (col1, col2) pairs to analyze.
            If None, analyzes all pairwise combinations.
        method : str
            'knn' or 'discretized'
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
            
            if len(x) < 50:  # NMI works with smaller samples
                mi = 0.0
            else:
                mi = self.compute_mi(x, y, method=method)
            
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
    
    def run_ariel_synthetic(self, n=1000, method='discretized', verbose=True):
        """
        Run NMI on Ariel's synthetic dataset.
        
        Parameters:
        -----------
        n : int
            Number of samples
        method : str
            'knn' or 'discretized'
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
            mi = self.compute_mi(x, y, method=method)
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
    print("NORMALIZED MI PIPELINE EXAMPLE")
    print("="*80)
    
    # Initialize pipeline
    pipeline = NMIPipeline(k=3)
    
    # Example 1: Run on Ariel's synthetic data
    print("\nExample 1: Ariel's Synthetic Dataset")
    print("-"*80)
    ariel_results = pipeline.run_ariel_synthetic(n=1000, method='discretized')
    print(f"\nDetection rate: {ariel_results['Detected'].sum()}/{len(ariel_results)}")
    
    # Example 2: Run on custom data
    print("\n" + "="*80)
    print("Example 2: Custom Dataset")
    print("-"*80)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'feature1': np.random.uniform(-2, 2, 500),
    })
    sample_data['feature2'] = sample_data['feature1']**2 + np.random.randn(500) * 0.3  # Quadratic
    sample_data['feature3'] = np.random.randn(500)  # Independent
    sample_data['feature4'] = sample_data['feature1'] * 2 + np.random.randn(500) * 0.1  # Linear
    
    custom_results = pipeline.run_dataset(sample_data, method='discretized')
    print("\nResults:")
    print(custom_results)
    
    # Save results
    ariel_results.to_csv('/mnt/user-data/outputs/pipeline_nmi_ariel_results.csv', index=False)
    custom_results.to_csv('/mnt/user-data/outputs/pipeline_nmi_custom_results.csv', index=False)
    
    print("\n" + "="*80)
    print("Results saved!")
    print("="*80)
