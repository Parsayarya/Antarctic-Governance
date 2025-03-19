import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.multitest import multipletests

def load_data(attributes_file, network_metrics_files):
    """
    Load attributes and network metrics data
    """
    # Load attributes
    attributes = pd.read_excel(attributes_file)
    
    # Load network metrics for each topic
    network_metrics = {}
    for topic_file in network_metrics_files:
        topic_num = topic_file.split('_')[1]
        network_metrics[f'topic_{topic_num}'] = pd.read_csv(topic_file)
    
    return attributes, network_metrics

def prepare_data(attributes_df, network_df):
    """
    Merge and prepare data for analysis
    """
    # Merge attributes with network metrics
    merged_df = pd.merge(attributes_df, network_df, 
                        left_on='Submitter', 
                        right_on='Author', 
                        how='inner')
    
    # Separate numerical and categorical columns
    numeric_cols = merged_df.select_dtypes(include=['float64', 'int64']).columns
    categorical_cols = merged_df.select_dtypes(include=['object']).columns
    
    # Remove 'Submitter' and 'Author' from analysis columns
    numeric_cols = numeric_cols[~numeric_cols.isin(['Submitter', 'Author'])]
    categorical_cols = categorical_cols[~categorical_cols.isin(['Submitter', 'Author'])]
    
    return merged_df, numeric_cols, categorical_cols

def permutation_test(data1, data2, n_permutations=10000):
    """
    Perform a permutation test to compare two groups
    
    Parameters:
    -----------
    data1, data2: arrays of data from two groups
    n_permutations: number of permutations to run
    
    Returns:
    --------
    p_value: p-value from permutation test
    effect_size: difference in means between groups
    """
    # Calculate the observed difference in means
    observed_diff = np.mean(data2) - np.mean(data1)
    
    # Combine the data
    combined_data = np.concatenate([data1, data2])
    n1, n2 = len(data1), len(data2)
    
    # Initialize counter for values more extreme than observed
    count_extreme = 0
    
    # Perform permutation test
    for _ in range(n_permutations):
        # Shuffle the combined data
        np.random.shuffle(combined_data)
        
        # Split into two groups of original sizes
        perm_data1 = combined_data[:n1]
        perm_data2 = combined_data[n1:n1+n2]
        
        # Calculate difference in means for permuted data
        perm_diff = np.mean(perm_data2) - np.mean(perm_data1)
        
        # Count if permutation difference is more extreme than observed
        if abs(perm_diff) >= abs(observed_diff):
            count_extreme += 1
    
    # Calculate p-value
    p_value = count_extreme / n_permutations
    
    # Calculate effect size (normalized difference)
    pooled_std = np.sqrt(((n1-1) * np.std(data1, ddof=1)**2 + 
                          (n2-1) * np.std(data2, ddof=1)**2) / 
                         (n1 + n2 - 2))
    
    if pooled_std == 0:
        effect_size = 0
    else:
        effect_size = observed_diff / pooled_std  # Cohen's d
    
    return p_value, effect_size

def analyze_binary_relationships_permutation(df, binary_cols, metric_cols, n_permutations=1000):
    """
    Analyze relationships between binary attributes and network metrics using permutation tests
    """
    results = []
    
    for binary_col in binary_cols:
        for metric_col in metric_cols:
            # Skip if too many null values
            if df[binary_col].isnull().sum() > len(df) * 0.5:
                continue
                
            # Get groups (0 and 1)
            group0 = df[df[binary_col] == 0][metric_col].dropna().values
            group1 = df[df[binary_col] == 1][metric_col].dropna().values
            
            if len(group0) > 0 and len(group1) > 0:
                # Perform permutation test
                p_value, effect_size = permutation_test(group0, group1, n_permutations)
                
                # Record the direction of effect (comparing group1 to group0)
                mean_diff = np.mean(group1) - np.mean(group0)
                direction = "Increases" if mean_diff > 0 else "Decreases"
                
                results.append({
                    'Attribute': binary_col,
                    'Network_Metric': metric_col,
                    'P_Value': p_value,
                    'Effect_Size': effect_size,
                    'Mean_Diff': mean_diff,
                    'Effect_Direction': direction,
                    'Group0_Mean': np.mean(group0),
                    'Group1_Mean': np.mean(group1),
                    'Group0_Size': len(group0),
                    'Group1_Size': len(group1)
                })
    
    return pd.DataFrame(results)

def permutation_test_categorical(groups, n_permutations=10000):
    """
    Perform a permutation test for categorical variables with multiple groups
    
    Parameters:
    -----------
    groups: list of arrays, each representing data from one category
    n_permutations: number of permutations to run
    
    Returns:
    --------
    p_value: p-value from permutation test
    """
    # Calculate the observed F-statistic (similar to ANOVA)
    group_means = [np.mean(group) for group in groups]
    group_sizes = [len(group) for group in groups]
    grand_mean = np.mean(np.concatenate(groups))
    
    # Between-group variability
    between_ss = sum([size * ((mean - grand_mean) ** 2) for size, mean in zip(group_sizes, group_means)])
    between_df = len(groups) - 1
    
    # Within-group variability
    within_ss = sum([sum((group - mean) ** 2) for group, mean in zip(groups, group_means)])
    within_df = sum(group_sizes) - len(groups)
    
    # Calculate F-statistic
    if within_df == 0 or within_ss == 0:
        observed_f = 0  # Handle edge case
    else:
        observed_f = (between_ss / between_df) / (within_ss / within_df)
    
    # Combine all data
    all_data = np.concatenate(groups)
    total_size = len(all_data)
    
    # Store group sizes for later splitting
    sizes = [len(group) for group in groups]
    
    # Perform permutation test
    count_extreme = 0
    
    for _ in range(n_permutations):
        # Shuffle all data
        np.random.shuffle(all_data)
        
        # Split data into groups of original sizes
        start_idx = 0
        perm_groups = []
        for size in sizes:
            perm_groups.append(all_data[start_idx:start_idx+size])
            start_idx += size
        
        # Calculate permuted F-statistic
        perm_means = [np.mean(group) for group in perm_groups]
        
        # Between-group variability
        perm_between_ss = sum([size * ((mean - grand_mean) ** 2) for size, mean in zip(sizes, perm_means)])
        
        # Within-group variability
        perm_within_ss = sum([sum((group - mean) ** 2) for group, mean in zip(perm_groups, perm_means)])
        
        # Calculate permuted F-statistic
        if perm_within_ss == 0:
            perm_f = 0
        else:
            perm_f = (perm_between_ss / between_df) / (perm_within_ss / within_df)
        
        # Count if permutation F is more extreme
        if perm_f >= observed_f:
            count_extreme += 1
    
    # Calculate p-value
    p_value = count_extreme / n_permutations
    
    return p_value, observed_f

def analyze_categorical_relationships_permutation(df, cat_cols, metric_cols, n_permutations=1000):
    """
    Analyze relationships between categorical attributes and network metrics using permutation tests
    """
    results = []
    
    for cat_col in cat_cols:
        for metric_col in metric_cols:
            # Skip if too many null values
            if df[cat_col].isnull().sum() > len(df) * 0.5:
                continue
            
            # Create groups and remove empty ones
            groups_dict = df.groupby(cat_col)[metric_col].apply(lambda x: x.dropna().values).to_dict()
            groups = [group for group in groups_dict.values() if len(group) > 0]
            categories = [cat for cat, group in groups_dict.items() if len(group) > 0]
            
            if len(groups) > 1:
                # Perform permutation test
                p_value, f_stat = permutation_test_categorical(groups, n_permutations)
                
                # Calculate means for each group
                group_means = {cat: np.mean(group) for cat, group in zip(categories, groups)}
                group_sizes = {cat: len(group) for cat, group in zip(categories, groups)}
                
                # Find category with max and min mean
                max_cat = max(categories, key=lambda k: group_means[k])
                min_cat = min(categories, key=lambda k: group_means[k])
                
                results.append({
                    'Attribute': cat_col,
                    'Network_Metric': metric_col,
                    'P_Value': p_value,
                    'F_Statistic': f_stat,
                    'Max_Category': max_cat,
                    'Max_Mean': group_means[max_cat],
                    'Min_Category': min_cat,
                    'Min_Mean': group_means[min_cat],
                    'Categories': categories,
                    'Category_Means': group_means,
                    'Category_Sizes': group_sizes
                })
    
    return pd.DataFrame(results)

def main():
    # Define file paths
    attributes_file = 'unique_submitters_attributes_GROUPS_v2.xlsx'
    network_files = [
        'topic_1_network_metrics_kneed.csv',
        'topic_5_network_metrics_kneed.csv',
        'topic_7_network_metrics_kneed.csv',
        'topic_14_network_metrics_kneed.csv'
    ]
    
    # Set number of permutations
    n_permutations = 10000
    
    # Load data
    attributes_df, network_metrics = load_data(attributes_file, network_files)
    
    # Store all results for final report
    all_binary_results = []
    all_categorical_results = []
    
    # Analyze each topic
    for topic_name, topic_df in network_metrics.items():
        print(f"\nAnalyzing {topic_name}")
        
        # Prepare data
        merged_df, numeric_cols, categorical_cols = prepare_data(attributes_df, topic_df)
        
        # Separate binary and non-binary numeric columns
        binary_cols = [col for col in numeric_cols 
                      if set(merged_df[col].dropna().unique()).issubset({0, 1})]
        metric_cols = [col for col in numeric_cols 
                      if col in topic_df.columns and col != 'Topic_Score']
        
        # Analyze binary relationships with permutation tests
        binary_results = analyze_binary_relationships_permutation(
            merged_df, binary_cols, metric_cols, n_permutations)
        
        # Analyze categorical relationships with permutation tests
        cat_results = analyze_categorical_relationships_permutation(
            merged_df, categorical_cols, metric_cols, n_permutations)
        
        # Apply multiple testing correction
        for results_df in [binary_results, cat_results]:
            if not results_df.empty:
                _, results_df['Adjusted_P_Value'], _, _ = multipletests(
                    results_df['P_Value'], method='fdr_bh')
        
        # Add topic information
        binary_results['Topic'] = topic_name
        cat_results['Topic'] = topic_name
        
        # Append to all results
        all_binary_results.append(binary_results)
        all_categorical_results.append(cat_results)
        
        # Print significant results
        print("\nSignificant binary relationships (adjusted p < 0.05):")
        significant_binary = binary_results[binary_results['Adjusted_P_Value'] < 0.05]
        if len(significant_binary) > 0:
            print(significant_binary[['Attribute', 'Network_Metric', 'P_Value', 
                                      'Adjusted_P_Value', 'Effect_Size', 
                                      'Effect_Direction', 'Mean_Diff']])
        else:
            print("None found.")
        
        print("\nSignificant categorical relationships (adjusted p < 0.05):")
        significant_cat = cat_results[cat_results['Adjusted_P_Value'] < 0.05]
        if len(significant_cat) > 0:
            print(significant_cat[['Attribute', 'Network_Metric', 'P_Value', 
                                  'Adjusted_P_Value', 'F_Statistic', 
                                  'Max_Category', 'Min_Category']])
        else:
            print("None found.")
    
    # Combine results across all topics
    if all_binary_results:
        all_binary_df = pd.concat(all_binary_results)
        all_binary_df.to_csv('all_binary_relationships_permutation.csv', index=False)
        print("\nSaved all binary results to all_binary_relationships_permutation.csv")
    
    if all_categorical_results:
        all_cat_df = pd.concat(all_categorical_results)
        all_cat_df.to_csv('all_categorical_relationships_permutation.csv', index=False)
        print("\nSaved all categorical results to all_categorical_relationships_permutation.csv")

if __name__ == "__main__":
    main()