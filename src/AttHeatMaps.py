import pandas as pd 

df = pd.read_excel('unique_submitters_attributes_GROUPS_v2.xlsx')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def create_correlation_heatmap(topic_n, df_metrics, df_attributes):
    # Merge the dataframes on Author/Submitter
    merged_df = df_metrics.merge(df_attributes, left_on='Author', right_on='Submitter', how='inner')
    
    # Drop unnecessary columns
    merged_df = merged_df.drop(['Author', 'Submitter', 'Type_actor'], axis=1)
    
    # Get metric columns
    metric_columns = ['Topic_Score', 'Weighted_Degree', 'Degree_Centrality', 
                     'Betweenness_Centrality', 'Closeness_Centrality', 'Eigenvector_Centrality']
    
    # Handle categorical variables
    categorical_columns = merged_df.select_dtypes(include=['object']).columns
    merged_df = pd.get_dummies(merged_df, columns=categorical_columns)
    
    # Drop rows with NaN values
    merged_df = merged_df.dropna()
    
    # Get all columns except metric columns for attributes
    attribute_columns = [col for col in merged_df.columns if col not in metric_columns]
    
    # Initialize correlation matrix with numpy array of floats
    correlation_matrix = np.zeros((len(attribute_columns), len(metric_columns)))
    
    # Calculate correlations with error handling
    for i, attr in enumerate(attribute_columns):
        attr_series = merged_df[attr]
        # Skip constant columns
        if attr_series.nunique() == 1:
            correlation_matrix[i, :] = np.nan
            continue
            
        for j, metric in enumerate(metric_columns):
            metric_series = merged_df[metric]
            # Skip if either series is constant
            if metric_series.nunique() == 1:
                correlation_matrix[i, j] = np.nan
                continue
                
            try:
                # Calculate correlation and handle potential warnings
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr = attr_series.corr(metric_series)
                correlation_matrix[i, j] = corr if not np.isnan(corr) else 0
            except Exception as e:
                print(f"Error calculating correlation between {attr} and {metric}: {e}")
                correlation_matrix[i, j] = np.nan
    
    # Convert to DataFrame for better handling
    correlation_matrix = pd.DataFrame(
        correlation_matrix,
        index=attribute_columns,
        columns=metric_columns
    )
    
    # Create heatmap
    plt.figure(figsize=(12, len(attribute_columns)//2))
    
    # Mask for NaN values
    mask = np.isnan(correlation_matrix)
    
    # Create heatmap with masked values
    sns.heatmap(correlation_matrix,
                mask=mask,
                cmap='RdBu',
                center=0,
                vmin=-1,
                vmax=1,
                annot=True,
                fmt='.2f',
                cbar_kws={'label': 'Correlation Coefficient'},
                square=True)
    
    plt.title(f'Correlation Heatmap for Topic {topic_n}')
    plt.xlabel('Network Metrics')
    plt.ylabel('Actor Attributes')
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save figure
    plt.savefig(f'correlation_heatmap_topic_{topic_n}.png', dpi=300, bbox_inches='tight')
    plt.close()

# Read the main attributes dataframe (assuming it's already loaded as df)
df_attributes = df.copy()

# Process each topic file
topic_numbers = [1, 5, 7, 14]
for topic_n in topic_numbers:
    # Read topic metrics file
    df_metrics = pd.read_csv(f'topic_{topic_n}_metrics.csv')
    
    # Create heatmap
    create_correlation_heatmap(topic_n, df_metrics, df_attributes)