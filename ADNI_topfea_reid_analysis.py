import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from reidentification_utils import find_closest_neighbors_Nyxus_topfea

def analyze_top_features_performance(ranking_key="permutation_importance"):
    """
    Analyze re-identification performance using different numbers of top features.
    """
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI_info.csv"
    base_output_dir = "./Nyxus_topfea_reid_analysis"
    # Load feature rankings
    features_rank_df = pd.read_csv("./train_disease_classifier/models/rf_feature_ranking_All_42.csv")

    if ranking_key == "rank":
        features_rank_df.sort_values(by=ranking_key, ascending=True, inplace=True)
    else:
        features_rank_df.sort_values(by=ranking_key, ascending=False, inplace=True)
    
    # Define feature numbers to test
    feature_nums = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150, 175, 210]
    
    # Initialize results storage
    results = {
        'num_features': [],
        'top1_rate': [],
        'top10_rate': [],
        'patient_top1_rate': [],
        'patient_top10_rate': [],
        'patient_ap': [],
        'image_ap': []
    }
    
    # Parameters for re-identification

    if not os.path.exists(base_output_dir):
        os.makedirs(base_output_dir)
    n_neighbors = 100
    standardize = True
    
    print("🔍 Starting Top Features Analysis...")
    print("=" * 60)
    
    # Test each number of features
    for num_feat in feature_nums:
        print(f"\n📊 Testing with top {num_feat} features...")
        
        # Get top features
        top_features = features_rank_df.head(num_feat)['feature'].tolist()
        
        # Create output directory for this test
        output_dir = f"./{base_output_dir}/top_{num_feat}_features"
        os.makedirs(output_dir, exist_ok=True)
        
        reid_results = find_closest_neighbors_Nyxus_topfea(
                features_dir=features_dir,
                image_dir=image_dir,
                info_csv=info_csv,
                features_names=top_features,
                n_neighbors=n_neighbors,
                standardize=standardize,
                num_top=num_feat,
                exclude_same_date=True,
                distance_threshold=-1.0,
                output_dir=output_dir
            )
            
        # Store results
        # import pdb; pdb.set_trace()
        results['num_features'].append(num_feat)
        results['top1_rate'].append(reid_results['r_at_1_img'])
        results['top10_rate'].append(reid_results['r_at_10_img'])
        results['patient_top1_rate'].append(reid_results['r_at_1_patient'])
        results['patient_top10_rate'].append(reid_results['r_at_10_patient'])
        results['patient_ap'].append(reid_results['patient_ap'])
        results['image_ap'].append(reid_results['image_ap'])

    # import pdb; pdb.set_trace()
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv(f"./{base_output_dir}/top_features_performance_{ranking_key}.csv", index=False)
    print(f"\n💾 Results saved to: ./Nyxus_topfea_analysis/top_features_performance_{ranking_key}.csv")
    
    return results_df


def create_summary_plot(results_df, base_output_dir, ranking_key):
    """
    Create a focused summary plot with the most important metrics.
    """
    
    # Convert percentages to decimals for consistency
    results_df = results_df.copy()
    results_df['top1_rate'] = results_df['top1_rate'] / 100.0 if results_df['top1_rate'].max() > 1 else results_df['top1_rate']
    results_df['patient_top1_rate'] = results_df['patient_top1_rate'] / 100.0 if results_df['patient_top1_rate'].max() > 1 else results_df['patient_top1_rate']
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Plot key metrics
    ax.plot(results_df['num_features'], results_df['top1_rate'], 
            marker='o', linewidth=3, markersize=10, label='Image Top-1 Rate', color='#e74c3c')
    ax.plot(results_df['num_features'], results_df['patient_top1_rate'], 
            marker='s', linewidth=3, markersize=10, label='Patient Top-1 Rate', color='#3498db')
    ax.plot(results_df['num_features'], results_df['image_ap'], 
            marker='d', linewidth=3, markersize=10, label='Image Average Precision', color='#9b59b6')
    ax.plot(results_df['num_features'], results_df['patient_ap'], 
            marker='^', linewidth=3, markersize=10, label='Patient Average Precision', color='#2ecc71')

    
    # Styling
    ax.set_xlabel('Number of Top Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Performance Score', fontsize=14, fontweight='bold')
    ax.set_title('Re-Identification Performance: Key Metrics vs Top Features', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add value annotations on data points
    for i, row in results_df.iterrows():
        ax.annotate(f'{row["top1_rate"]:.2f}', 
                   (row['num_features'], row['top1_rate']), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{row["patient_top1_rate"]:.2f}', 
                   (row['num_features'], row['patient_top1_rate']), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
        ax.annotate(f'{row["patient_ap"]:.2f}', 
                   (row['num_features'], row['patient_ap']), 
                   textcoords="offset points", xytext=(0,10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f"./{base_output_dir}/summary_performance_{ranking_key}_plot.png", 
                dpi=300, bbox_inches='tight')
    
    print(f"📈 Summary plot saved to: ./{base_output_dir}/summary_performance_{ranking_key}_plot.png")


def print_summary_table(results_df, ranking_key):
    """
    Print a formatted summary table of results.
    """
    print("\n" + "="*80)
    print("🏆 TOP FEATURES PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Features':<8} {'Top-1':<8} {'Top-10':<8} {'Pat-T1':<8} {'Pat-T10':<8} {'Pat-AP':<8} {'Img-AP':<8}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        print(f"{int(row['num_features']):<8} "
              f"{row['top1_rate']:<8.3f} "
              f"{row['top10_rate']:<8.3f} "
              f"{row['patient_top1_rate']:<8.3f} "
              f"{row['patient_top10_rate']:<8.3f} "
              f"{row['patient_ap']:<8.3f} "
              f"{row['image_ap']:<8.3f}")
    
    # Find best performing configurations
    print(f"\n🎯 BEST CONFIGURATIONS {ranking_key}:")
    print("-" * 40)
    
    best_top1_idx = results_df['top1_rate'].idxmax()
    best_patient_ap_idx = results_df['patient_ap'].idxmax()
    best_image_ap_idx = results_df['image_ap'].idxmax()
    
    combined_score = results_df['top1_rate']
    best_combined_idx = combined_score.idxmax()
    
    print(f"Best Top-1 Rate: {int(results_df.loc[best_top1_idx, 'num_features'])} features "
          f"({results_df.loc[best_top1_idx, 'top1_rate']:.3f})")
    print(f"Best Image AP: {int(results_df.loc[best_image_ap_idx, 'num_features'])} features "
          f"({results_df.loc[best_image_ap_idx, 'image_ap']:.3f})")
    print(f"Best Patient AP: {int(results_df.loc[best_patient_ap_idx, 'num_features'])} features "
          f"({results_df.loc[best_patient_ap_idx, 'patient_ap']:.3f})")

    print(f"Best Combined:   {int(results_df.loc[best_combined_idx, 'num_features'])} features "
          f"({combined_score.loc[best_combined_idx]:.3f})")


def save_best_features(ranking_key, num_top):
    """
    Save the best features to a file.
    """
    image_dir = "/home/ubuntu/data/ADNI_dataset/BrainIAC_processed/images/"
    features_dir = "/home/ubuntu/data/ADNI_dataset/Nyxus_features/"
    info_csv = "/home/ubuntu/data/ADNI_dataset/BrainIAC_input_csv/brainiac_ADNI_info.csv"
    base_output_dir = "./Nyxus_topfea_reid_analysis"
    # Load feature rankings
    features_rank_df = pd.read_csv("./train_disease_classifier/models/rf_feature_ranking_All_42.csv")

    if ranking_key == "rank":
        features_rank_df.sort_values(by=ranking_key, ascending=True, inplace=True)
    else:
        features_rank_df.sort_values(by=ranking_key, ascending=False, inplace=True)

    # Fix: Use double brackets for multiple column selection
    best_features_df = features_rank_df[['feature', f'{ranking_key}']]
    best_features_df.to_csv(f"./{base_output_dir}/best_features_{ranking_key}_{num_top}.csv", index=False)
    
    print(f"✅ Saved top {num_top} features based on {ranking_key} to:")
    print(f"   {base_output_dir}/best_features_{ranking_key}_{num_top}.csv")


        

if __name__ == "__main__":
    # Run the analysis
    base_output_dir = "./Nyxus_topfea_reid_analysis"
    print("🚀 Starting Top Features Re-Identification Analysis...")
    
    # Check if results already exist, otherwise run analysis
    #results_file = f"./{base_output_dir}/top_features_performance.csv"
    # for ranking_key in ["rank", "drop_column_importance"]:
    #     results_df = pd.read_csv(f"./{base_output_dir}/top_features_performance_{ranking_key}.csv")
        
    #     #results_df = analyze_top_features_performance(ranking_key=ranking_key)
        
    #     # Create plots
    #     print("📈 Creating performance plots...")
    #     #create_summary_plot(results_df, base_output_dir, ranking_key=ranking_key)
        
    #     # Print summary
    #     print_summary_table(results_df, ranking_key=ranking_key)
        
    #     print("\n✅ Analysis complete!")
    #     print(f"📁 Check './{base_output_dir}/' for all results and plots") 

    save_best_features(ranking_key="rank", num_top=25)