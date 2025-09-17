"""
t-SNE Clustering and Visualization for ADNI Classification
Analyze feature separability and clustering quality for CN, MCI, AD classes.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, normalized_mutual_info_score,
    homogeneity_score, completeness_score, v_measure_score,
    calinski_harabasz_score, davies_bouldin_score
)
from sklearn.neighbors import NearestNeighbors
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from calssifier_utils import prepare_features_Nyxus, prepare_features_BrainAIC

RANDOM_STATE = 123

class TSNEClassificationAnalyzer:
    """
    Comprehensive t-SNE analysis for ADNI classification data.
    """
    
    def __init__(self, random_state: int = RANDOM_STATE):
        self.random_state = random_state
        self.tsne_results = None
        self.pca_results = None
        self.clustering_results = {}
        
    def prepare_data_for_tsne(self, X: np.ndarray, y: np.ndarray, 
                             patient_ids: np.ndarray = None,
                             apply_pca_first: bool = True,
                             pca_components: int = 50) -> tuple:
        """
        Prepare data for t-SNE analysis with optional PCA preprocessing.
        
        Args:
            X: Feature matrix
            y: Labels
            patient_ids: Patient IDs (optional)
            apply_pca_first: Whether to apply PCA before t-SNE
            pca_components: Number of PCA components
            
        Returns:
            Processed data ready for t-SNE
        """
        print(f"Preparing data for t-SNE analysis...")
        print(f"Original data shape: {X.shape}")
        print(f"Classes: {np.unique(y)} (counts: {np.bincount(y)})")
        
        # Standardize features
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)
        X_scaled = X
        
        # Optional PCA preprocessing (recommended for high-dimensional data)
        if apply_pca_first and X.shape[1] > pca_components:
            print(f"Applying PCA preprocessing ({pca_components} components)...")
            pca = PCA(n_components=pca_components, random_state=self.random_state)
            X_pca = pca.fit_transform(X_scaled)
            
            # Store PCA results
            self.pca_results = {
                'pca_object': pca,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
                'n_components': pca_components
            }
            
            print(f"PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f}")
            X_processed = X_pca
        else:
            X_processed = X_scaled
            
        print(f"Final data shape for t-SNE: {X_processed.shape}")
        
        return X_processed, y, patient_ids
    
    def run_tsne_analysis(self, X: np.ndarray, y: np.ndarray,
                         perplexity_values: list = [30, 50, 100],
                         learning_rate_values: list = [200, 500, 1000],
                         max_iter: int = 1000) -> dict:
        """
        Run t-SNE with different parameter combinations.
        
        Args:
            X: Preprocessed feature matrix
            y: Labels
            perplexity_values: List of perplexity values to try
            learning_rate_values: List of learning rates to try
            max_iter: Maximum number of iterations (renamed from n_iter)
            
        Returns:
            Dictionary with t-SNE results for different parameters
        """
        print(f"Running t-SNE analysis with multiple parameter combinations...")
        
        tsne_results = {}
        
        for perplexity in perplexity_values:
            for learning_rate in learning_rate_values:
                param_key = f"perp_{perplexity}_lr_{learning_rate}"
                print(f"  Running t-SNE: perplexity={perplexity}, learning_rate={learning_rate}")
                
                # Adjust perplexity if it's too large for the dataset
                max_perplexity = min(perplexity, (len(X) - 1) // 3)
                
                tsne = TSNE(
                    n_components=2,
                    perplexity=max_perplexity,
                    learning_rate=learning_rate,
                    max_iter=max_iter,  # Changed from n_iter to max_iter
                    random_state=self.random_state,
                    verbose=0,
                    metric='euclidean',
                    init='pca'
                )
                
                try:
                    embedding = tsne.fit_transform(X)
                    
                    # Calculate clustering quality metrics
                    silhouette = silhouette_score(embedding, y)
                    calinski_harabasz = calinski_harabasz_score(embedding, y)
                    davies_bouldin = davies_bouldin_score(embedding, y)
                    
                    tsne_results[param_key] = {
                        'embedding': embedding,
                        'perplexity': max_perplexity,
                        'learning_rate': learning_rate,
                        'kl_divergence': tsne.kl_divergence_,
                        'silhouette_score': silhouette,
                        'calinski_harabasz_score': calinski_harabasz,
                        'davies_bouldin_score': davies_bouldin,
                        'tsne_object': tsne
                    }
                    
                    print(f"    KL divergence: {tsne.kl_divergence_:.3f}")
                    print(f"    Silhouette score: {silhouette:.3f}")
                    
                except Exception as e:
                    print(f"    Error: {str(e)}")
                    continue
        
        self.tsne_results = tsne_results
        return tsne_results
    
    def find_best_tsne_parameters(self) -> tuple:
        """
        Find the best t-SNE parameters based on multiple criteria.
        
        Returns:
            Best parameter key and results
        """
        if not self.tsne_results:
            raise ValueError("No t-SNE results available. Run run_tsne_analysis first.")
        
        print("Finding best t-SNE parameters...")
        
        # Score each parameter combination
        scores = {}
        for param_key, result in self.tsne_results.items():
            # Combine multiple metrics (lower KL divergence and Davies-Bouldin, higher Silhouette and Calinski-Harabasz)
            normalized_kl = 1 / (1 + result['kl_divergence'])  # Lower is better
            normalized_db = 1 / (1 + result['davies_bouldin_score'])  # Lower is better
            normalized_sil = result['silhouette_score']  # Higher is better
            normalized_ch = result['calinski_harabasz_score'] / 1000  # Higher is better, normalize
            
            # Combined score
            combined_score = (normalized_kl + normalized_db + normalized_sil + normalized_ch) / 4
            scores[param_key] = combined_score
            
            print(f"  {param_key}: combined_score={combined_score:.3f}")
        
        # Find best parameters
        best_param_key = max(scores.keys(), key=lambda x: scores[x])
        best_result = self.tsne_results[best_param_key]
        
        print(f"\nBest parameters: {best_param_key}")
        print(f"  Perplexity: {best_result['perplexity']}")
        print(f"  Learning rate: {best_result['learning_rate']}")
        print(f"  Combined score: {scores[best_param_key]:.3f}")
        
        return best_param_key, best_result
    
    def plot_tsne_results(self, y: np.ndarray, class_names: list,
                         patient_ids: np.ndarray = None,
                         plot_type: str = 'all',
                         save_plots: bool = True) -> None:
        """
        Create comprehensive t-SNE visualizations.
        
        Args:
            y: True labels
            class_names: Names of classes
            patient_ids: Patient IDs (optional)
            plot_type: 'best', 'comparison', 'all'
            save_plots: Whether to save plots
        """
        if not self.tsne_results:
            raise ValueError("No t-SNE results available. Run run_tsne_analysis first.")
        
        print("Creating t-SNE visualizations...")
        
        if plot_type in ['best', 'all']:
            self._plot_best_tsne(y, class_names, patient_ids, save_plots)
        
        if plot_type in ['comparison', 'all']:
            self._plot_tsne_comparison(y, class_names, save_plots)
        
        # if plot_type in ['interactive', 'all']:
        #     self._plot_interactive_tsne(y, class_names, patient_ids, save_plots)
    
    def _plot_best_tsne(self, y: np.ndarray, class_names: list,
                       patient_ids: np.ndarray = None, save_plots: bool = True):
        """Plot the best t-SNE result."""
        best_param_key, best_result = self.find_best_tsne_parameters()
        embedding = best_result['embedding']
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Plot 1: By class
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, class_name in enumerate(class_names):
            mask = y == i
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=colors[i], label=class_name, alpha=0.7, s=30)
        
        axes[0].set_title(f'Best t-SNE: Classes\n{best_param_key}', fontsize=14)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Density plot
        axes[1].hexbin(embedding[:, 0], embedding[:, 1], gridsize=30, cmap='YlOrRd', alpha=0.7)
        axes[1].set_title('t-SNE Density Plot', fontsize=14)
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        
        # Add metrics as text
        metrics_text = f"Silhouette: {best_result['silhouette_score']:.3f}\n"
        metrics_text += f"Calinski-Harabasz: {best_result['calinski_harabasz_score']:.1f}\n"
        metrics_text += f"Davies-Bouldin: {best_result['davies_bouldin_score']:.3f}\n"
        metrics_text += f"KL Divergence: {best_result['kl_divergence']:.3f}"
        
        axes[1].text(0.02, 0.98, metrics_text, transform=axes[1].transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('./plots/tsne_best_result.png', dpi=300, bbox_inches='tight')
            print("Best t-SNE plot saved to: ./plots/tsne_best_result.png")
        
        plt.show()
    
    def _plot_tsne_comparison(self, y: np.ndarray, class_names: list, save_plots: bool = True):
        """Plot comparison of different t-SNE parameters."""
        n_results = len(self.tsne_results)
        n_cols = min(3, n_results)
        n_rows = (n_results + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for idx, (param_key, result) in enumerate(self.tsne_results.items()):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            embedding = result['embedding']
            
            for i, class_name in enumerate(class_names):
                mask = y == i
                ax.scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=colors[i], label=class_name if idx == 0 else "", alpha=0.7, s=20)
            
            ax.set_title(f'{param_key}\nSil: {result["silhouette_score"]:.3f}', fontsize=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_results, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)
        
        # Add legend
        if n_results > 0:
            axes[0, 0].legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('./plots/tsne_parameter_comparison.png', dpi=300, bbox_inches='tight')
            print("t-SNE comparison plot saved to: ./plots/tsne_parameter_comparison.png")
        
        plt.show()
    
    def _plot_interactive_tsne(self, y: np.ndarray, class_names: list,
                              patient_ids: np.ndarray = None, save_plots: bool = True):
        """Create interactive t-SNE plot using Plotly (with fallback)."""
        try:
            import plotly.express as px
            
            best_param_key, best_result = self.find_best_tsne_parameters()
            embedding = best_result['embedding']
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'tsne_1': embedding[:, 0],
                'tsne_2': embedding[:, 1],
                'class': [class_names[i] for i in y],
                'class_num': y
            })
            
            if patient_ids is not None:
                df['patient_id'] = patient_ids
            
            # Create interactive plot
            fig = px.scatter(
                df, x='tsne_1', y='tsne_2', color='class',
                title=f'Interactive t-SNE Visualization - {best_param_key}',
                hover_data=['patient_id'] if patient_ids is not None else None,
                color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            
            fig.update_layout(
                width=800, height=600,
                xaxis_title='t-SNE Component 1',
                yaxis_title='t-SNE Component 2'
            )
            
            if save_plots:
                fig.write_html('./plots/tsne_interactive.html')
                print("Interactive t-SNE plot saved to: ./plots/tsne_interactive.html")
            
            fig.show()
            
        except ImportError:
            print("Plotly not available. Skipping interactive plot.")
            print("Install plotly with: pip install plotly")
        except Exception as e:
            print(f"Error creating interactive plot: {str(e)}")
            print("Continuing without interactive visualization...")
    
    def run_clustering_analysis(self, y: np.ndarray, class_names: list) -> dict:
        """
        Run clustering analysis on the best t-SNE embedding.
        
        Args:
            y: True labels
            class_names: Names of classes
            
        Returns:
            Dictionary with clustering results
        """
        if not self.tsne_results:
            raise ValueError("No t-SNE results available. Run run_tsne_analysis first.")
        
        print("Running clustering analysis on t-SNE embedding...")
        
        best_param_key, best_result = self.find_best_tsne_parameters()
        embedding = best_result['embedding']
        
        clustering_results = {}
        
        # K-Means clustering
        print("  Running K-Means clustering...")
        kmeans = KMeans(n_clusters=len(class_names), random_state=self.random_state)
        kmeans_labels = kmeans.fit_predict(embedding)
        
        clustering_results['kmeans'] = {
            'labels': kmeans_labels,
            'centers': kmeans.cluster_centers_,
            'inertia': kmeans.inertia_,
            'adjusted_rand_score': adjusted_rand_score(y, kmeans_labels),
            'normalized_mutual_info': normalized_mutual_info_score(y, kmeans_labels),
            'homogeneity': homogeneity_score(y, kmeans_labels),
            'completeness': completeness_score(y, kmeans_labels),
            'v_measure': v_measure_score(y, kmeans_labels)
        }
        
        # DBSCAN clustering
        print("  Running DBSCAN clustering...")
        # Estimate eps parameter
        neighbors = NearestNeighbors(n_neighbors=4)
        neighbors_fit = neighbors.fit(embedding)
        distances, indices = neighbors_fit.kneighbors(embedding)
        distances = np.sort(distances[:, 3], axis=0)
        eps = np.percentile(distances, 90)  # Use 90th percentile as eps
        
        dbscan = DBSCAN(eps=eps, min_samples=5)
        dbscan_labels = dbscan.fit_predict(embedding)
        
        if len(np.unique(dbscan_labels)) > 1:  # Check if clustering was successful
            clustering_results['dbscan'] = {
                'labels': dbscan_labels,
                'eps': eps,
                'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                'n_noise': list(dbscan_labels).count(-1),
                'adjusted_rand_score': adjusted_rand_score(y, dbscan_labels),
                'normalized_mutual_info': normalized_mutual_info_score(y, dbscan_labels),
                'homogeneity': homogeneity_score(y, dbscan_labels),
                'completeness': completeness_score(y, dbscan_labels),
                'v_measure': v_measure_score(y, dbscan_labels)
            }
        
        # Hierarchical clustering
        print("  Running Hierarchical clustering...")
        hierarchical = AgglomerativeClustering(n_clusters=len(class_names))
        hierarchical_labels = hierarchical.fit_predict(embedding)
        
        clustering_results['hierarchical'] = {
            'labels': hierarchical_labels,
            'adjusted_rand_score': adjusted_rand_score(y, hierarchical_labels),
            'normalized_mutual_info': normalized_mutual_info_score(y, hierarchical_labels),
            'homogeneity': homogeneity_score(y, hierarchical_labels),
            'completeness': completeness_score(y, hierarchical_labels),
            'v_measure': v_measure_score(y, hierarchical_labels)
        }
        
        self.clustering_results = clustering_results
        return clustering_results
    
    def plot_clustering_results(self, y: np.ndarray, class_names: list, save_plots: bool = True):
        """
        Plot clustering results comparison.
        
        Args:
            y: True labels
            class_names: Names of classes
            save_plots: Whether to save plots
        """
        if not self.clustering_results:
            raise ValueError("No clustering results available. Run run_clustering_analysis first.")
        
        best_param_key, best_result = self.find_best_tsne_parameters()
        embedding = best_result['embedding']
        
        n_methods = len(self.clustering_results) + 1  # +1 for true labels
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
        
        if n_methods == 1:
            axes = [axes]
        
        # Plot true labels
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for i, class_name in enumerate(class_names):
            mask = y == i
            axes[0].scatter(embedding[mask, 0], embedding[mask, 1], 
                          c=colors[i], label=class_name, alpha=0.7, s=30)
        
        axes[0].set_title('True Labels', fontsize=12)
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot clustering results
        for idx, (method, result) in enumerate(self.clustering_results.items(), 1):
            cluster_labels = result['labels']
            n_clusters = len(np.unique(cluster_labels))
            
            # Use different colors for each cluster
            scatter = axes[idx].scatter(embedding[:, 0], embedding[:, 1], 
                                      c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
            
            # Add cluster centers for K-means
            if method == 'kmeans' and 'centers' in result:
                axes[idx].scatter(result['centers'][:, 0], result['centers'][:, 1], 
                                c='red', marker='x', s=100, linewidths=3)
            
            axes[idx].set_title(f'{method.title()}\nARI: {result["adjusted_rand_score"]:.3f}', fontsize=12)
            axes[idx].set_xlabel('t-SNE 1')
            axes[idx].set_ylabel('t-SNE 2')
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('./plots/clustering_comparison.png', dpi=300, bbox_inches='tight')
            print("Clustering comparison plot saved to: ./plots/clustering_comparison.png")
        
        plt.show()
    
    def generate_clustering_report(self, class_names: list) -> str:
        """
        Generate comprehensive clustering analysis report.
        
        Args:
            class_names: Names of classes
            
        Returns:
            String containing the full report
        """
        if not self.tsne_results or not self.clustering_results:
            raise ValueError("Missing results. Run analysis first.")
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("COMPREHENSIVE t-SNE CLUSTERING ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append("")
        
        # t-SNE Analysis Summary
        best_param_key, best_result = self.find_best_tsne_parameters()
        report_lines.append("t-SNE ANALYSIS SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"Best parameters: {best_param_key}")
        report_lines.append(f"  Perplexity: {best_result['perplexity']}")
        report_lines.append(f"  Learning rate: {best_result['learning_rate']}")
        report_lines.append(f"  KL divergence: {best_result['kl_divergence']:.4f}")
        report_lines.append(f"  Silhouette score: {best_result['silhouette_score']:.4f}")
        report_lines.append(f"  Calinski-Harabasz score: {best_result['calinski_harabasz_score']:.2f}")
        report_lines.append(f"  Davies-Bouldin score: {best_result['davies_bouldin_score']:.4f}")
        report_lines.append("")
        
        # PCA Summary (if applied)
        if self.pca_results:
            report_lines.append("PCA PREPROCESSING SUMMARY:")
            report_lines.append("-" * 30)
            report_lines.append(f"Components used: {self.pca_results['n_components']}")
            report_lines.append(f"Explained variance: {np.sum(self.pca_results['explained_variance_ratio']):.4f}")
            report_lines.append(f"Top 5 components variance: {self.pca_results['explained_variance_ratio'][:5]}")
            report_lines.append("")
        
        # Clustering Results Summary
        report_lines.append("CLUSTERING ANALYSIS RESULTS:")
        report_lines.append("-" * 40)
        
        for method, result in self.clustering_results.items():
            report_lines.append(f"{method.upper()} Clustering:")
            report_lines.append(f"  Adjusted Rand Index: {result['adjusted_rand_score']:.4f}")
            report_lines.append(f"  Normalized Mutual Info: {result['normalized_mutual_info']:.4f}")
            report_lines.append(f"  Homogeneity: {result['homogeneity']:.4f}")
            report_lines.append(f"  Completeness: {result['completeness']:.4f}")
            report_lines.append(f"  V-measure: {result['v_measure']:.4f}")
            
            if method == 'kmeans':
                report_lines.append(f"  Inertia: {result['inertia']:.2f}")
            elif method == 'dbscan':
                report_lines.append(f"  Number of clusters: {result['n_clusters']}")
                report_lines.append(f"  Noise points: {result['n_noise']}")
                report_lines.append(f"  Eps parameter: {result['eps']:.4f}")
            
            report_lines.append("")
        
        # Best clustering method
        best_clustering_method = max(self.clustering_results.keys(), 
                                   key=lambda x: self.clustering_results[x]['adjusted_rand_score'])
        best_ari = self.clustering_results[best_clustering_method]['adjusted_rand_score']
        
        report_lines.append("CLUSTERING PERFORMANCE SUMMARY:")
        report_lines.append("-" * 40)
        report_lines.append(f"Best clustering method: {best_clustering_method.upper()}")
        report_lines.append(f"Best Adjusted Rand Index: {best_ari:.4f}")
        
        # Interpretation
        report_lines.append("")
        report_lines.append("INTERPRETATION:")
        report_lines.append("-" * 20)
        if best_ari > 0.7:
            report_lines.append("EXCELLENT: Classes are very well separated in t-SNE space")
        elif best_ari > 0.5:
            report_lines.append("GOOD: Classes show good separation in t-SNE space")
        elif best_ari > 0.3:
            report_lines.append("MODERATE: Classes show some separation in t-SNE space")
        else:
            report_lines.append("POOR: Classes are not well separated in t-SNE space")
        
        report_lines.append("")
        report_lines.append("="*80)
        
        full_report = "\n".join(report_lines)
        return full_report

def analyze_adni_tsne_BrainAIC_clustering():
    """
    Complete t-SNE clustering analysis for ADNI data.
    """
    print("="*80)
    print("COMPREHENSIVE t-SNE CLUSTERING ANALYSIS FOR ADNI DATA")
    print("="*80)
    
    # Load data
    features_csv_path = "/home/ubuntu/data/ADNI_dataset/BrainIAC_features/features.csv"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    # Prepare data
    print("Preparing and splitting data")
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_BrainAIC(
        features_csv_path, info_csv, test_size=0.2, val_size=0.15
    )
    
    
    # Combine all data for analysis
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    patient_ids_all = np.concatenate([patient_ids_train, patient_ids_val, patient_ids_test], axis=0)
    
    print(f"Total data for analysis: {X_all.shape}")
    print(f"Class distribution: {np.bincount(y_all)}")
    
    # Create analyzer
    analyzer = TSNEClassificationAnalyzer(random_state=RANDOM_STATE)
    
    # Prepare data
    X_processed, y_processed, patient_ids_processed = analyzer.prepare_data_for_tsne(
        X_all, y_all, patient_ids_all, apply_pca_first=True, pca_components=50
    )
    
    # Run t-SNE analysis with corrected parameter name
    tsne_results = analyzer.run_tsne_analysis(
        X_processed, y_processed,
        perplexity_values=[20, 30, 50],
        learning_rate_values=[200, 500],
        max_iter=1000  # Changed from n_iter to max_iter
    )
    
    # Create plots directory
    import os
    os.makedirs('./plots', exist_ok=True)
    
    # Plot t-SNE results
    analyzer.plot_tsne_results(y_processed, class_names, patient_ids_processed, 
                              plot_type='all', save_plots=True)
    
    # Run clustering analysis   
    clustering_results = analyzer.run_clustering_analysis(y_processed, class_names)
    
    # Plot clustering results
    analyzer.plot_clustering_results(y_processed, class_names, save_plots=True)
    
    # Generate report
    report = analyzer.generate_clustering_report(class_names)
    
    # Save report
    with open('./plots/tsne_clustering_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print(f"\nAnalysis complete! All plots and report saved to ./plots/")
    
    return analyzer, tsne_results, clustering_results

def analyze_adni_tsne_Nyxus_clustering():
    """
    Complete t-SNE clustering analysis for ADNI data.
    """
    print("="*80)
    print("COMPREHENSIVE t-SNE CLUSTERING ANALYSIS FOR ADNI DATA")
    print("="*80)
    
    # Load data
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/ADNI1_Complete_3Yr_1.5T_diagonosis.xlsx"
    
    X_train, X_val, X_test, y_train, y_val, y_test, class_names, patient_ids_train, patient_ids_val, patient_ids_test = prepare_features_Nyxus(
        features_dir, image_dir, info_csv, test_size=0.2, val_size=0.15, features_group="All", classes=['CN','AD', 'MCI']
    )
    
    # Combine all data for analysis
    X_all = np.concatenate([X_train, X_val, X_test], axis=0)
    y_all = np.concatenate([y_train, y_val, y_test], axis=0)
    patient_ids_all = np.concatenate([patient_ids_train, patient_ids_val, patient_ids_test], axis=0)
    
    print(f"Total data for analysis: {X_all.shape}")
    print(f"Class distribution: {np.bincount(y_all)}")
    
    # Create analyzer
    analyzer = TSNEClassificationAnalyzer(random_state=RANDOM_STATE)
    
    # Prepare data
    X_processed, y_processed, patient_ids_processed = analyzer.prepare_data_for_tsne(
        X_all, y_all, patient_ids_all, apply_pca_first=True, pca_components=50
    )
    
    # Run t-SNE analysis with corrected parameter name
    tsne_results = analyzer.run_tsne_analysis(
        X_processed, y_processed,
        perplexity_values=[20, 30, 50],
        learning_rate_values=[200, 500],
        max_iter=1000  # Changed from n_iter to max_iter
    )
    
    # Create plots directory
    import os
    os.makedirs('./plots', exist_ok=True)
    
    # Plot t-SNE results
    #import ipdb; ipdb.set_trace()
    analyzer.plot_tsne_results(y_processed, class_names, patient_ids_processed, 
                              plot_type='all', save_plots=True)
    
    # Run clustering analysis
    clustering_results = analyzer.run_clustering_analysis(y_processed, class_names)
    
    # Plot clustering results
    analyzer.plot_clustering_results(y_processed, class_names, save_plots=True)
    
    # Generate report
    report = analyzer.generate_clustering_report(class_names)
    
    # Save report
    with open('./plots/tsne_clustering_report.txt', 'w') as f:
        f.write(report)
    
    print("\n" + report)
    print(f"\nAnalysis complete! All plots and report saved to ./plots/")
    
    return analyzer, tsne_results, clustering_results

if __name__ == "__main__":
    # analyzer, tsne_results, clustering_results = analyze_adni_tsne_Nyxus_clustering()   
    analyzer, tsne_results, clustering_results = analyze_adni_tsne_BrainAIC_clustering()
