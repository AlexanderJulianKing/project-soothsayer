"""
Exploratory Data Analysis for LLM Benchmarks.

This module provides comprehensive EDA capabilities for analyzing relationships
between different LLM benchmark scores. It includes:

- **Correlation Analysis**: Pairwise correlations with significance testing
- **Dimensionality Reduction**: PCA, t-SNE, and optional UMAP visualizations
- **Clustering**: K-means clustering with silhouette scoring and profiling
- **Visualization**: Heatmaps, scatter plots, radar charts for cluster profiles

The analysis helps identify:
1. Which benchmarks are redundant (highly correlated)
2. Natural groupings of models (clusters)
3. Latent structure in the benchmark space (principal components)
4. Outliers and unusual model performance patterns

Configuration:
    MISSING_THRESHOLD: Maximum allowed missing value ratio for model inclusion.
        Models with more missing values are excluded from analysis.
    USE_UMAP: Whether to use UMAP for visualization (requires umap-learn).
    INCLUDE_MISSINGNESS_FLAGS: Toggle creation of __was_missing indicator columns
        before imputation.

Example Usage:
    >>> df = read_and_clean_data('combined_all_benches.csv')
    >>> analysis_df, models = prepare_data_for_analysis(df)
    >>> corr_matrix = analyze_correlations(analysis_df)
    >>> pca_results, variance = perform_pca(analysis_df)
    >>> cluster_labels = perform_clustering(pca_results)

Dependencies:
    Required: pandas, numpy, matplotlib, seaborn, scikit-learn
    Optional: umap-learn (for UMAP visualization)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import math
from scipy.stats import ConstantInputWarning
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

USE_UMAP = False
MISSING_THRESHOLD = 0.508  # Max missing ratio for model inclusion
INCLUDE_MISSINGNESS_FLAGS = False  # Set False to skip adding __was_missing indicator columns
EXCLUDE_MODELS = {
    "Mistral 7B Instruct",  # Arena ELO 1110 — extreme outlier (z=-4.7)
}
# MISSING_THRESHOLD = 0.35
# Try to import UMAP, but make it optional
try:
    import umap
    USE_UMAP = True
    pass
except ImportError:
    print("UMAP could not be imported. UMAP visualization will be skipped.")
except Exception as e:
    print(f"Error importing UMAP: {e}. UMAP visualization will be skipped.")

# Helper to drop missing-flag columns from downstream analysis
def drop_missing_flag_columns(df):
    return df.loc[:, ~df.columns.str.endswith("__was_missing")]

# ==============================================================================
# DATA LOADING AND CLEANING
# ==============================================================================

def read_and_clean_data(file_path):
    """Read and clean LLM benchmarks data from CSV.

    Performs the following cleaning steps:
    1. Identify the model name column (tries Unified_Name, simplebench_Model_Mapped, etc.)
    2. Convert percentage strings to floats (e.g., "75%" -> 75.0)
    3. Attempt numeric conversion on object columns
    4. Drop completely empty rows

    Args:
        file_path: Path to the CSV file containing benchmark data.

    Returns:
        DataFrame with 'model_name' column and cleaned numeric columns.

    Note:
        Columns in skip_columns list are not converted to numeric to preserve
        text data like organization names and URLs.
    """
    # Read the CSV file
    df = pd.read_csv(file_path)
    # Create a column with model names for easier identification
    model_column_candidates = ['Unified_Name', 'simplebench_Model_Mapped', 'simplebench_Model']
    
    for col in model_column_candidates:
        if col in df.columns and df[col].notna().any():
            df['model_name'] = df[col]
            print(f"Using '{col}' as model name column.")
            break
    else:
        # Default to the first column containing 'Model' in its name
        model_cols = [col for col in df.columns if 'Model' in col]
        if model_cols:
            df['model_name'] = df[model_cols[0]]
        else:
            print("Warning: No model name column found. Creating generic model names.")
            df['model_name'] = [f"Model_{i}" for i in range(len(df))]

    # Skip columns to ignore in numerical conversion
    skip_columns = [
        'model_name', 'simplebench_Model', 'simplebench_Organization', 'simplebench_Model_Mapped',
        'Unified_Name', 'livebench_Model', 'livebench_Organization', 'livebench_Reasoning Average', 
        'livebench_Coding Average', 'livebench_Mathematics Average', 'livebench_Data Analysis Average', 
        'livebench_Language Average', 'livebench_IF Average', 'openbench_Model',
        'openbench_Source', 'hallucination_Model', 'hallucination_Model_Mapped', 'lmsys_model',
        'lmsys_organization', 'lmsys_license', 'lmsys_knowledge_cutoff', 'lmsys_url',
        'lmsys_Model', 'lmsys_Model_Mapped', 'lechmazur_Model', 'lechmazur_Model_Mapped',
        'lechmazur_Skipped/Total', 'livebench_Unnamed: 0', 'aider_Model_Mapped', 'lmsys_95_pct_ci', 'aider_Model', 'aider_Edit format',
        'aider_Command', 'ugileaderboard_Model', 'ugileaderboard_Model_Mapped_to_OpenBench',
        'aagdpval_Model', 'aagdpval_Creator', 'aagdpval_isReasoning', 'aagdpval_Model_Mapped_to_OpenBench',
        'aaomniscience_Model', 'aaomniscience_Creator', 'aaomniscience_isReasoning', 'aaomniscience_Model_Mapped_to_OpenBench',
        'aacritpt_Model', 'aacritpt_Creator', 'aacritpt_isReasoning', 'aacritpt_Model_Mapped_to_OpenBench'
    ]
    # Automatically process all other columns
    for col in df.columns:
        if col in skip_columns or df[col].dtype == 'bool':
            continue
            
        if df[col].dtype == 'object':
            # Check if it looks like URLs, text or other non-numeric data
            sample_vals = df[col].dropna().astype(str).iloc[:10] if len(df[col].dropna()) > 0 else []
            if any('http' in str(val).lower() for val in sample_vals) or len(sample_vals) == 0:
                continue
                
            # Process percentage values
            if any('%' in str(val) for val in sample_vals):
                try:
                    df[col] = df[col].astype(str).str.replace('%', '').astype(float)
                    print(f"Converted percentage column: {col}")
                except Exception:
                    pass  # Skip if conversion fails
            else:
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    if not df[col].isna().all():  # Check if conversion yielded any non-NA values
                        print(f"Converted to numeric: {col}")
                except:
                    pass
    
    # Drop rows that are completely empty or have no meaningful data
    df = df.dropna(how='all')

    
    # Filter out non-model entries if we have rank information
    # rank_cols = [col for col in df.columns if 'Rank' in col ]
    # if rank_cols:
    #     for col in rank_cols:
    #         if df[col].notna().any():
    #             df = df[df[col].notna()]
    #             break

    # print(df['model_name'])
    # print(fnskjfns)
    return df



# ==============================================================================
# DATA PREPARATION FOR ANALYSIS
# ==============================================================================

def prepare_data_for_analysis(df, exclude_columns=None, threshold_margin=0.2, include_missingness_flags=INCLUDE_MISSINGNESS_FLAGS):
    """Prepare benchmark data for correlation and clustering analysis.

    This function performs critical data preparation steps:
    1. Select numeric columns (excluding ranks and vote counts)
    2. Filter models by missing value threshold (MISSING_THRESHOLD)
    3. Remove zero-variance columns
    4. Optionally create missingness indicators for imputation tracking
    5. Impute missing values with column medians (including indicators if enabled)

    Args:
        df: DataFrame from read_and_clean_data() with 'model_name' column.
        exclude_columns: List of column names to exclude from analysis.
        threshold_margin: Window around threshold to report "close" models.
            Defaults to 0.2 (20%).
        include_missingness_flags: Whether to add __was_missing indicator columns
            before imputation. Defaults to INCLUDE_MISSINGNESS_FLAGS.

    Returns:
        Tuple of (analysis_df, model_names) where:
        - analysis_df: Cleaned DataFrame indexed by model_name with imputed values
        - model_names: Array of model names that passed the missing threshold

    Side Effects:
        Prints diagnostic information about:
        - Models that nearly made/missed the threshold
        - Number of models/columns excluded
        - Column-level missingness patterns

    Note:
        Models with > MISSING_THRESHOLD fraction of missing values are excluded.
        Columns with > col_completeness_threshold missing are excluded.
    """
    # Select only numerical columns
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    # Exclude rank columns and vote counts
    exclude_patterns = ['rank', 'Rank', 'votes', 'Skipped']
    numeric_cols = [col for col in numeric_cols if col == "lechmazur_gen_Avg Rank" or not any(pattern in col for pattern in exclude_patterns)]
    
    # Exclude specific columns if provided
    if exclude_columns:
        numeric_cols = [col for col in numeric_cols if col not in exclude_columns]
    print(f"Excluding {len(exclude_columns)} configured columns.")

    # Create analysis dataframe with model names as index
    analysis_df = df[['model_name'] + numeric_cols].set_index('model_name')
    print(f"{len(analysis_df.columns)} numeric columns, {len(analysis_df)} models.")

    # Calculate missing value ratio for each model
    missing_ratio = analysis_df.isna().mean(axis=1)
    
    # Identify threshold and models that are close but don't make the cut
    missing_threshold = MISSING_THRESHOLD
    col_completeness_threshold = 0.28
    almost_included = (missing_ratio >= missing_threshold) & (missing_ratio < missing_threshold + threshold_margin)
    almost_NOTincluded = (missing_ratio <= missing_threshold) & (missing_ratio > missing_threshold - threshold_margin)

    print(f"Missing-value threshold: {missing_threshold:.3f}")
    if almost_included.any():
        names = list(missing_ratio[almost_included].sort_values(ascending=True).index)
        print(f"  {len(names)} model(s) narrowly excluded: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")
    if almost_NOTincluded.any():
        names = list(missing_ratio[almost_NOTincluded].sort_values(ascending=False).index)
        print(f"  {len(names)} model(s) narrowly included: {', '.join(names[:5])}{'...' if len(names) > 5 else ''}")

        
    
    # Exclude manually blacklisted models (extreme outliers, etc.)
    if EXCLUDE_MODELS:
        excluded = analysis_df.index.isin(EXCLUDE_MODELS)
        if excluded.any():
            print(f"Excluding {excluded.sum()} blacklisted model(s): {', '.join(analysis_df.index[excluded])}")
            analysis_df = analysis_df[~excluded]

    # Filter rows based on threshold
    rows_to_keep = missing_ratio[analysis_df.index] < missing_threshold
    print(f"Removing {(~rows_to_keep).sum()} models with more than {missing_threshold*100}% missing values")
    analysis_df = analysis_df[rows_to_keep]
    


    # Remove columns with single value or no variance
    cols_with_variance = [col for col in analysis_df.columns if analysis_df[col].nunique() > 1 and analysis_df[col].var() > 0]


    analysis_df = analysis_df[cols_with_variance]
    print(f"{len(cols_with_variance)} columns after removing zero-variance.")

    # Compute column completeness on DENSE models only (<=50.8% missing)
    # so that adding sparse models doesn't cause useful columns to be dropped
    DENSE_THRESHOLD_FOR_COLS = 0.508
    # Recompute missing ratio on the filtered analysis_df to avoid reindex issues with duplicate model names
    filtered_missing_ratio = analysis_df.isna().mean(axis=1)
    dense_rows = filtered_missing_ratio < DENSE_THRESHOLD_FOR_COLS
    if dense_rows.any():
        non_na_ratio = analysis_df.loc[dense_rows].notna().mean(axis=0)
    else:
        non_na_ratio = analysis_df.notna().mean(axis=0)
    cols_to_keep_by_completeness = non_na_ratio >= col_completeness_threshold

    cols_to_remove = analysis_df.columns[~cols_to_keep_by_completeness]
    if not cols_to_remove.empty:
        print(f"Removing {len(cols_to_remove)} columns with <{col_completeness_threshold:.0%} non-NA values (computed on {dense_rows.sum()} dense models).")

    analysis_df = analysis_df.loc[:, cols_to_keep_by_completeness]



    analysis_df_non_imputed = analysis_df.copy()
    analysis_df_with_flags = analysis_df

    if include_missingness_flags:
        # Create missingness indicator columns for any feature that has at least one NaN
        missing_indicator_df = analysis_df.isna().astype(int)
        # Keep only indicators with at least 2 missing and 2 non-missing values (drop singletons/near-constants)
        indicator_counts = missing_indicator_df.sum(axis=0)
        min_pos = 5
        valid_indicator_mask = (indicator_counts >= min_pos) & (indicator_counts <= (len(analysis_df) - min_pos))
        missing_indicator_df = missing_indicator_df.loc[:, valid_indicator_mask]
        if missing_indicator_df.empty:
            print("No missingness indicator columns added (no mixed missingness after filtering).")
        else:
            missing_indicator_df.columns = [f"{col}__was_missing" for col in missing_indicator_df.columns]
            print(f"Adding {len(missing_indicator_df.columns)} missingness indicator columns.")
            # Store a non-imputed copy (including missingness flags) for exporting/analysis
            analysis_df_non_imputed = pd.concat([analysis_df, missing_indicator_df], axis=1)
            analysis_df_with_flags = pd.concat([analysis_df, missing_indicator_df], axis=1)
    else:
        print("Skipping missingness indicator columns (INCLUDE_MISSINGNESS_FLAGS=False).")
    
    # Perform imputation using IterativeImputer instead of simple mean imputation
    from sklearn.experimental import enable_iterative_imputer  # noqa
    from sklearn.impute import IterativeImputer
    iter_imputer = IterativeImputer(random_state=42)
    analysis_df = pd.DataFrame(iter_imputer.fit_transform(analysis_df_with_flags), 
                               columns=analysis_df_with_flags.columns, 
                               index=analysis_df_with_flags.index)
    
    return analysis_df, analysis_df_non_imputed

# ==============================================================================
# CORRELATION ANALYSIS
# ==============================================================================

def analyze_correlations(df_non_imputed):
    """Calculate and visualize pairwise correlations between benchmark metrics.

    Computes Pearson correlation coefficients using pairwise-complete observations
    (for each pair of columns, only rows with valid data for both are used).
    This avoids the need for imputation while maximizing data usage.

    Args:
        df_non_imputed: DataFrame with benchmark scores. Missing values are OK.
            Missingness flag columns (ending with '__was_missing') are auto-excluded.

    Returns:
        Tuple of (correlation_matrix, high_corr_pairs) where:
        - correlation_matrix: Full NxN correlation matrix as DataFrame
        - high_corr_pairs: DataFrame of pairs with |correlation| > 0.95

    Outputs:
        - analysis_output/correlation_matrix.csv: Full correlation matrix
        - analysis_output/correlation_heatmap.png: Visual heatmap

    Note:
        Low-variance/constant columns are automatically excluded to avoid
        undefined correlations and divide-by-zero warnings.
    """
    # Explicitly remove missingness-flag columns from correlation analysis
    flag_cols = [c for c in df_non_imputed.columns if str(c).endswith("__was_missing")]
    if flag_cols:
        print(f"Skipping {len(flag_cols)} missingness-flag column(s) in correlation matrix: {flag_cols}")
        df_non_imputed = df_non_imputed.drop(columns=flag_cols, errors='ignore')

    def _is_low_variance(col: pd.Series, dominance_thresh: float = 0.90, min_std: float = 1e-8, min_minority: int = 2) -> bool:
        s = col.dropna()
        if len(s) <= 1:
            return True
        if s.std(ddof=0) < min_std:
            return True
        vc = s.value_counts(normalize=True)
        if vc.empty:
            return True
        minority_count = len(s) - int(s.value_counts().iloc[0])
        if vc.iloc[0] >= dominance_thresh or minority_count < min_minority:
            return True
        return False

    # Drop constant columns to avoid undefined correlations / warnings
    drop_cols = [c for c in df_non_imputed.columns if _is_low_variance(df_non_imputed[c])]
    if drop_cols:
        print(f"Skipping {len(drop_cols)} low-variance/constant columns in correlation matrix: {drop_cols}")
    df_corr = df_non_imputed.drop(columns=drop_cols, errors='ignore')

    # Calculate correlation matrix. The .corr() method automatically handles
    # missing values by using only pairwise-complete observations.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=ConstantInputWarning)
        correlation_matrix = df_corr.corr(method='pearson') # 'pearson' is default

    # Save correlation matrix as CSV for downstream analysis
    os.makedirs('analysis_output', exist_ok=True)
    correlation_matrix.to_csv('analysis_output/correlation_matrix.csv')
    
    # Plot correlation heatmap
    plt.figure(figsize=(18, 16))
    # Note: The heatmap will show the full matrix, but each cell was calculated pairwise.
    ax = sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap='coolwarm',
        fmt='.2f',
        linewidths=0.5,
        annot_kws={"size": 2}
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=6)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=6)
    ax.tick_params(axis='x', labelbottom=True, labeltop=False)
    plt.title('Correlation Matrix of LLM Metrics (No Imputation)', fontsize=10)
    plt.tight_layout()
    plt.savefig('analysis_output/correlation_heatmap.png', dpi=300)
    plt.close()
    
    # Identify highly correlated features
    high_corr_features = {}
    # Make a copy to avoid modifying the original matrix while iterating
    corr_matrix_upper = correlation_matrix.copy()
    # Keep only the upper triangle to avoid duplicate pairs
    corr_matrix_upper.loc[:,:] = np.triu(corr_matrix_upper.values, k=1)
    
    # Find pairs with high correlation
    high_corr_pairs = corr_matrix_upper.stack().reset_index()
    high_corr_pairs.columns = ['Feature 1', 'Feature 2', 'Correlation']
    high_corr_pairs = high_corr_pairs[abs(high_corr_pairs['Correlation']) > 0.95]
    
    # Sort by absolute correlation value
    high_corr_pairs['Abs_Correlation'] = high_corr_pairs['Correlation'].abs()
    sorted_corr = high_corr_pairs.sort_values(by='Abs_Correlation', ascending=False).drop('Abs_Correlation', axis=1)

    return correlation_matrix, sorted_corr

# ==============================================================================
# DIMENSIONALITY REDUCTION
# ==============================================================================

def perform_pca(df):
    """Perform Principal Component Analysis on benchmark data.

    PCA identifies the principal components (directions of maximum variance)
    in the high-dimensional benchmark space. This helps understand:
    1. How many independent factors underlie the benchmarks
    2. Which benchmarks contribute to each factor
    3. How to visualize models in reduced dimensions

    Args:
        df: DataFrame with numeric benchmark scores (imputed, no missing values).

    Returns:
        Tuple of (pca_df, pca_model) where:
        - pca_df: DataFrame with PC1, PC2, PC3 coordinates for each model
        - pca_model: Fitted PCA object with loadings and variance info

    Outputs:
        - analysis_output/pca_explained_variance.png: Scree plot
        - analysis_output/pca_loadings.csv: Feature loadings per component
        - Prints top contributing features for PC1-PC3
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Determine optimal number of components
    pca = PCA()
    pca.fit(scaled_data)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Plot explained variance
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-')
    plt.axhline(y=0.8, color='r', linestyle='--', label='80% Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA Explained Variance')
    plt.legend()
    plt.grid(True)
    plt.savefig('analysis_output/pca_explained_variance.png', dpi=300)
    plt.close()
    
    # Get number of components for 80% variance
    n_components = next((i for i, v in enumerate(cumulative_variance) if v >= 0.8), len(df.columns)-1) + 1
    
    # Use at least 2 components for visualization
    n_components = max(2, min(n_components, len(df.columns)))
    
    # Perform PCA with optimal components
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create DataFrame with principal components
    pca_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df.index
    )
    
    # Get feature importance for each component
    feature_importance = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=df.columns
    )
    
    return pca_df, pca.explained_variance_ratio_, feature_importance

def perform_tsne(df):
    """Perform t-SNE dimensionality reduction for visualization.

    t-SNE (t-distributed Stochastic Neighbor Embedding) is a nonlinear
    dimensionality reduction technique that preserves local structure.
    Unlike PCA, it can reveal clusters and nonlinear relationships.

    Args:
        df: DataFrame with numeric benchmark scores (imputed, no missing values).

    Returns:
        DataFrame with t-SNE1 and t-SNE2 coordinates for each model.

    Note:
        Perplexity is automatically adjusted for small datasets to avoid errors.
        Results are sensitive to perplexity; may need tuning for different data.
    """
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply t-SNE for 2D visualization
    # Set perplexity to min(5, number of samples - 1) to handle small datasets
    perplexity = min(5, max(1, len(df) - 1))
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(scaled_data)
    
    # Create DataFrame with t-SNE results
    tsne_df = pd.DataFrame(
        data=tsne_results,
        columns=['t-SNE1', 't-SNE2'],
        index=df.index
    )
    
    return tsne_df

# Function to perform UMAP (only called if UMAP is available)
def perform_umap(df):
    """Perform UMAP dimensionality reduction"""
    if not USE_UMAP:
        return None
        
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Apply UMAP with adjusted parameters for small datasets
    n_neighbors = min(5, max(2, len(df) - 1))
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, random_state=42)
    umap_results = reducer.fit_transform(scaled_data)
    
    # Create DataFrame with UMAP results
    umap_df = pd.DataFrame(
        data=umap_results,
        columns=['UMAP1', 'UMAP2'],
        index=df.index
    )
    
    return umap_df

# Function to visualize dimensionality reduction results
def visualize_reduced_dimensions(pca_df, tsne_df, umap_df=None, cluster_df=None, original_df=None): # Add original_df here
    """Visualize the results of dimensionality reduction techniques"""
    # Set up a color palette based on company/organization if available
    company_col = None
    company_colors = None
    model_to_company = None
    unique_companies = None

    if original_df is not None:
        # Try to find a suitable organization column
        org_col_candidates = ['simplebench_Organization', 'lmsys_organization', 'livebench_Organization', 'openbench_Source']
        for col in org_col_candidates:
            if col in original_df.columns:
                company_col = col
                print(f"Using '{company_col}' for coloring.")
                break

        if company_col:
            # Create a mapping from model_name to company, handling potential missing values
            # Ensure original_df is indexed correctly if needed (assuming 'model_name' column exists)
            if 'model_name' in original_df.columns:
                 # Handle potential duplicate model names by taking the first entry
                original_df_indexed = original_df.drop_duplicates(subset=['model_name']).set_index('model_name')
                # Map model names from pca_df index to company using the indexed original_df
                model_to_company = pca_df.index.map(original_df_indexed[company_col].fillna('Unknown'))
            else:
                 print("Warning: 'model_name' column not found in original_df for company mapping.")
                 company_col = None # Fallback to no company coloring

            if model_to_company is not None:
                unique_companies = model_to_company.unique()
                # Use a palette suitable for many categories
                palette = sns.color_palette("hls", len(unique_companies))
                company_colors = {company: palette[i] for i, company in enumerate(unique_companies)}
        else:
            print("Warning: No suitable company/organization column found. Plotting without company colors.")

    # Function to create a scatter plot with model annotations
    def create_scatter_plot(data, x_col, y_col, title, filename):
        plt.figure(figsize=(12, 10)) # Increased size slightly

        # Color by company if possible, otherwise default or cluster
        if company_colors and model_to_company is not None:
            # Add company info to the data being plotted for easier filtering
            data_with_company = data.copy()
            # Ensure alignment - only map companies for models present in 'data'
            data_with_company['company'] = data.index.map(original_df.set_index('model_name')[company_col].fillna('Unknown'))

            for company in unique_companies:
                models_in_company = data_with_company[data_with_company['company'] == company]
                if not models_in_company.empty:
                    plt.scatter(
                        models_in_company[x_col],
                        models_in_company[y_col],
                        label=company,
                        color=company_colors[company],
                        alpha=0.8,
                        s=50 # Slightly larger points
                    )
            legend_title = 'Company/Organization'

        # Fallback: Color by cluster if company info is unavailable but clusters exist
        elif cluster_df is not None:
            print("Company info not used for coloring, falling back to clusters.")
            n_clusters = cluster_df['Cluster'].nunique()
            cluster_palette = sns.color_palette("tab10", n_clusters)
            cluster_colors_map = {i: cluster_palette[i] for i in range(n_clusters)}

            # Ensure cluster_df index aligns with data index before mapping
            aligned_clusters = cluster_df.reindex(data.index)['Cluster']

            for cluster_id in sorted(aligned_clusters.dropna().unique()):
                 # Get indices directly from the aligned series
                 models_in_cluster_idx = aligned_clusters[aligned_clusters == cluster_id].index
                 subset = data.loc[models_in_cluster_idx]
                 plt.scatter(
                     subset[x_col],
                     subset[y_col],
                     label=f'Cluster {cluster_id}',
                     color=cluster_colors_map[cluster_id],
                     alpha=0.7,
                     s=50
                 )
            legend_title = 'Cluster'
        # Fallback: No colors
        else:
            print("No company or cluster info for coloring. Using default color.")
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=50)
            legend_title = None

        # Annotate points with model names - adjust to prevent overlap
        texts = []
        for i, model in enumerate(data.index):
            texts.append(plt.text(data.iloc[i, 0], data.iloc[i, 1], model, fontsize=8)) # Smaller font

        # Attempt to adjust text labels to reduce overlap (requires adjustText library)
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            print("Consider installing 'adjustText' (`pip install adjustText`) for better label placement.")
            pass # Continue without adjustment if library not installed

        plt.title(title, fontsize=16)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        if legend_title:
             # Place legend outside the plot
             plt.legend(title=legend_title, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
        plt.savefig(filename, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
        plt.close()

    # PCA Visualization
    create_scatter_plot(
        pca_df, 'PC1', 'PC2',
        f'PCA colored by {"Company" if company_col else "Default"}', # Dynamic title
        'analysis_output/pca_visualization_by_company.png' # New filename
    )

    # t-SNE Visualization (Optional: you could apply the same coloring logic here)
    # Keeping t-SNE and UMAP with cluster coloring for comparison, unless you want to change them too.
    # You would call create_scatter_plot similarly for tsne_df and umap_df if needed.
    # If you want t-SNE also colored by company:
    # create_scatter_plot(
    #     tsne_df, 't-SNE1', 't-SNE2',
    #     f't-SNE colored by {"Company" if company_col else "Default"}',
    #     'analysis_output/tsne_visualization_by_company.png'
    # )
    # Else, keep the original t-SNE/UMAP plots (which use cluster coloring if available):
    def create_scatter_plot_original_coloring(data, x_col, y_col, title, filename):
        plt.figure(figsize=(12, 10))
        color_applied = False
        if cluster_df is not None:
            # Align cluster_df index with data index
            aligned_clusters = cluster_df.reindex(data.index)['Cluster']
            if not aligned_clusters.isnull().all(): # Check if any clusters were assigned
                n_clusters = int(aligned_clusters.max() + 1) # Safely get number of clusters
                palette = sns.color_palette("tab10", n_clusters)
                cluster_colors = {i: palette[i] for i in range(n_clusters)}

                for cluster_id in sorted(aligned_clusters.dropna().unique()):
                    models_in_cluster_idx = aligned_clusters[aligned_clusters == cluster_id].index
                    subset = data.loc[models_in_cluster_idx]
                    plt.scatter(
                        subset[x_col],
                        subset[y_col],
                        label=f'Cluster {cluster_id}',
                        color=cluster_colors[int(cluster_id)], # Ensure cluster_id is int
                        alpha=0.7,
                        s=50
                    )
                plt.legend(title='Clusters', bbox_to_anchor=(1.05, 1), loc='upper left')
                color_applied = True

        if not color_applied:
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=50)

        texts = []
        for i, model in enumerate(data.index):
            texts.append(plt.text(data.iloc[i, 0], data.iloc[i, 1], model, fontsize=8))

        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
             pass

        plt.title(title)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(True)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

    create_scatter_plot_original_coloring(
        tsne_df, 't-SNE1', 't-SNE2',
        't-SNE Visualization (Colored by Cluster if available)',
        'analysis_output/tsne_visualization_by_cluster.png' # Keep original or rename
    )

    if umap_df is not None:
        create_scatter_plot_original_coloring(
            umap_df, 'UMAP1', 'UMAP2',
            'UMAP Visualization (Colored by Cluster if available)',
            'analysis_output/umap_visualization_by_cluster.png' # Keep original or rename
        )


# ==============================================================================
# CLUSTERING ANALYSIS
# ==============================================================================

def perform_clustering(df, n_clusters_range=None):
    """Perform K-means clustering with automatic cluster count selection.

    Uses silhouette score to determine the optimal number of clusters.
    Silhouette score measures how similar models are to their own cluster
    vs. other clusters (higher = better separation).

    Args:
        df: DataFrame with benchmark scores or PCA coordinates.
        n_clusters_range: Tuple of (min, max) clusters to try.
            Defaults to (2, min(10, n_models-1)).

    Returns:
        Tuple of (cluster_labels, cluster_analysis) where:
        - cluster_labels: Series mapping model names to cluster IDs
        - cluster_analysis: DataFrame with cluster profiles (mean values)

    Outputs:
        - analysis_output/silhouette_scores.png: Plot of silhouette vs. k
        - analysis_output/cluster_profiles.csv: Mean values per cluster
        - Prints optimal k and silhouette scores

    Note:
        Clustering is performed on standardized data. Results may vary with
        random seed; set random_state for reproducibility.
    """
    # Set default cluster range
    if n_clusters_range is None:
        max_clusters = min(5, len(df) - 1)
        n_clusters_range = range(1, max_clusters + 1) if max_clusters >= 2 else [2]
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    for n_clusters in n_clusters_range:
        if n_clusters < len(df):  # Ensure we have enough data points
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(scaled_data)
            silhouette_avg = silhouette_score(scaled_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
    
    if not silhouette_scores:  # If we couldn't compute any scores
        optimal_clusters = min(2, len(df))
    else:
        # Get optimal number of clusters
        optimal_clusters = n_clusters_range[np.argmax(silhouette_scores)]
    
    # Perform clustering with optimal number
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42, n_init=10)
    df_clustered = df.copy()
    df_clustered['Cluster'] = kmeans.fit_predict(scaled_data)
    
    # Plot silhouette scores if we have more than one cluster option
    if len(silhouette_scores) > 1:
        plt.figure(figsize=(10, 6))
        plt.plot(n_clusters_range[:len(silhouette_scores)], silhouette_scores, marker='o')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Silhouette Score')
        plt.title('Optimal Number of Clusters')
        plt.grid(True)
        plt.savefig('analysis_output/silhouette_scores.png', dpi=300)
        plt.close()
    
    return df_clustered, optimal_clusters

# Function to visualize dimensionality reduction results
def visualize_reduced_dimensions(pca_df, tsne_df, umap_df=None, cluster_df=None, original_df=None): # Add original_df here
    """Visualize the results of dimensionality reduction techniques with company colors and shapes."""
    # Set up variables
    company_col = None
    company_colors = None
    company_markers = None
    model_to_company = None
    unique_companies = None

    if original_df is not None:
        # Try to find a suitable organization column - Prioritize openbench_Source as requested
        org_col_candidates = ['openbench_Source', 'simplebench_Organization', 'lmsys_organization', 'livebench_Organization']
        for col in org_col_candidates:
            if col in original_df.columns and original_df[col].notna().any(): # Check if col exists and has data
                company_col = col
                print(f"Using '{company_col}' for coloring and shapes.")
                break

        if company_col:
            # Create a mapping from model_name to company, filling NaNs with 'Unknown'
            if 'model_name' in original_df.columns:
                # Ensure unique model names in the index for mapping
                original_df_indexed = original_df.drop_duplicates(subset=['model_name']).set_index('model_name')
                # Map index of pca_df (which should be model names) to the company column
                model_to_company = pca_df.index.map(original_df_indexed[company_col].fillna('Unknown'))
                # Check if mapping was successful (sometimes index alignment issues occur)
                if model_to_company.isnull().all():
                     print(f"Warning: Mapping model names to company column '{company_col}' resulted in all NaNs. Check index alignment.")
                     company_col = None # Fallback
                else:
                    unique_companies = sorted(model_to_company.unique())
            else:
                 print("Warning: 'model_name' column not found in original_df for company mapping.")
                 company_col = None # Fallback

            # --- Color and Marker Assignment ---
            if company_col and unique_companies:
                has_unknown = 'Unknown' in unique_companies
                companies_for_palette = [c for c in unique_companies if c != 'Unknown']

                # 1. Assign Colors (Gray for Unknown)
                palette = sns.color_palette("hls", len(companies_for_palette))
                company_colors = {company: palette[i] for i, company in enumerate(companies_for_palette)}
                if has_unknown:
                    company_colors['Unknown'] = 'gray' # Assign gray to Unknown

                # 2. Assign Markers (Unique for top N, default for rest/Unknown)
                # Define available markers and how many unique ones to use
                markers_list = ['o', 's', '^', 'D', 'v', 'P', '*', 'X', 'p', '<', '>'] # More markers
                num_unique_shapes = min(len(markers_list), len(companies_for_palette)) # Use up to N unique shapes for known companies
                default_marker = '.' # Small dot for less frequent companies and 'Unknown'

                # Find the most frequent known companies
                company_counts = model_to_company[model_to_company != 'Unknown'].value_counts()
                top_companies = company_counts.head(num_unique_shapes).index.tolist()

                company_markers = {}
                marker_idx = 0
                for company in unique_companies:
                    if company == 'Unknown':
                        company_markers[company] = default_marker
                    elif company in top_companies:
                        company_markers[company] = markers_list[marker_idx]
                        marker_idx += 1
                    else: # Assign default marker to less frequent known companies
                        company_markers[company] = default_marker
        else:
            print("Warning: No suitable company/organization column found or mapping failed. Plotting without company colors/shapes.")

    # --- Nested Scatter Plot Function ---
    def create_scatter_plot(data, x_col, y_col, title, filename):
        plt.figure(figsize=(14, 11)) # Slightly larger figure

        # Determine coloring/shaping method
        use_company_styling = company_colors and company_markers and model_to_company is not None
        use_cluster_coloring = cluster_df is not None and not use_company_styling

        legend_title = None

        if use_company_styling:
            legend_title = 'Company/Organization'
            # Add company info to the data being plotted for easier filtering
            # Remap here to ensure alignment with potentially filtered 'data' index
            current_model_to_company = data.index.map(original_df.drop_duplicates(subset=['model_name']).set_index('model_name')[company_col].fillna('Unknown'))
            data_with_company = data.copy()
            data_with_company['company'] = current_model_to_company

            for company in unique_companies:
                models_in_company = data_with_company[data_with_company['company'] == company]
                if not models_in_company.empty:
                    plt.scatter(
                        models_in_company[x_col],
                        models_in_company[y_col],
                        label=company,
                        color=company_colors.get(company, 'black'), # Use .get for safety
                        marker=company_markers.get(company, 'o'), # Use .get for safety
                        alpha=0.8,
                        s=60 # Slightly larger points for better shape visibility
                    )

        elif use_cluster_coloring:
            legend_title = 'Cluster'
            print("Company info not used for styling, falling back to clusters for color.")
            # Align cluster_df index with data index
            aligned_clusters = cluster_df.reindex(data.index)['Cluster']
            if not aligned_clusters.isnull().all():
                # Use integer cluster IDs if possible, handle potential floats/strings carefully
                try:
                    unique_cluster_ids = sorted(aligned_clusters.dropna().astype(int).unique())
                    n_clusters = len(unique_cluster_ids)
                except ValueError: # Handle cases where cluster IDs might not be simple integers
                     unique_cluster_ids = sorted(aligned_clusters.dropna().unique())
                     n_clusters = len(unique_cluster_ids)
                     print(f"Warning: Cluster IDs seem non-integer: {unique_cluster_ids}")

                cluster_palette = sns.color_palette("tab10", n_clusters)
                # Map potentially non-integer IDs to palette indices
                cluster_colors_map = {cid: cluster_palette[i % len(cluster_palette)] for i, cid in enumerate(unique_cluster_ids)}


                for cluster_id in unique_cluster_ids:
                    models_in_cluster_idx = aligned_clusters[aligned_clusters == cluster_id].index
                    subset = data.loc[models_in_cluster_idx]
                    plt.scatter(
                        subset[x_col],
                        subset[y_col],
                        label=f'Cluster {cluster_id}',
                        color=cluster_colors_map[cluster_id],
                        marker='o', # Default marker for clusters
                        alpha=0.7,
                        s=50
                    )
            else:
                 print("No cluster assignments found for coloring.")
                 plt.scatter(data[x_col], data[y_col], alpha=0.7, s=50, marker='o')

        else: # Fallback: No colors or shapes
            print("No company or cluster info for styling. Using default color and shape.")
            plt.scatter(data[x_col], data[y_col], alpha=0.7, s=50, marker='o')


        # Annotate points with model names - adjust to prevent overlap
        texts = []
        for i, model in enumerate(data.index):
             # Check if model exists in the original index before trying to plot text
             if model in data.index:
                 model_data = data.loc[model]
                 texts.append(plt.text(model_data[x_col], model_data[y_col], model, fontsize=8))

        # Attempt to adjust text labels
        try:
            from adjustText import adjust_text
            if texts: # Only run if there are texts to adjust
                 adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
        except ImportError:
            print("Consider installing 'adjustText' (`pip install adjustText`) for better label placement.")
            pass # Continue without adjustment

        plt.title(title, fontsize=16)
        plt.xlabel(x_col, fontsize=12)
        plt.ylabel(y_col, fontsize=12)
        if legend_title:
             # Place legend outside the plot, adjust based on number of items
             num_legend_items = len(unique_companies) if use_company_styling else (len(unique_cluster_ids) if use_cluster_coloring and not aligned_clusters.isnull().all() else 0)
             # Simple heuristic to potentially use more columns for many items
             ncol_legend = 2 if num_legend_items > 15 else 1
             plt.legend(title=legend_title, bbox_to_anchor=(1.04, 1), loc='upper left', ncol=ncol_legend, fontsize=9)

        plt.grid(True)
        # Adjust layout AFTER legend placement attempt
        plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend (may need tuning)
        plt.savefig(filename, dpi=300, bbox_inches='tight') # Use bbox_inches='tight'
        plt.close()

    # --- Generate Plots ---

    # PCA Visualization (Company Colors/Shapes)
    pca_title = f'PCA colored by {company_col}' if company_col and company_colors else 'PCA (Default Styling)'
    create_scatter_plot(
        pca_df, 'PC1', 'PC2',
        pca_title,
        'analysis_output/pca_visualization_by_company_shape.png' # New filename
    )

    # t-SNE Visualization (Cluster Colors - default 'o' marker)
    # (You could adapt create_scatter_plot call to use company styling for t-SNE too if desired)
    create_scatter_plot(
        tsne_df, 't-SNE1', 't-SNE2',
        't-SNE Visualization (Colored by Cluster if available)',
        'analysis_output/tsne_visualization_by_cluster.png' # Keeping cluster coloring here
    )

    # UMAP Visualization (Cluster Colors - default 'o' marker)
    if umap_df is not None:
        create_scatter_plot(
            umap_df, 'UMAP1', 'UMAP2',
            'UMAP Visualization (Colored by Cluster if available)',
            'analysis_output/umap_visualization_by_cluster.png' # Keeping cluster coloring here
        )

def list_nearly_complete_models(df, max_missing_count=2):
    # Count missing values per model (assuming model names are the index)
    missing_counts = df.isna().sum(axis=1)
    nearly_complete = missing_counts[missing_counts <= max_missing_count]
    
    print(f"\nModels with {max_missing_count} or fewer missing values:")
    for model, count in nearly_complete.items():
        print(f"  - {model}: {count} missing values")
    return nearly_complete

# Function to analyze cluster characteristics
def analyze_clusters(clustered_df, original_df):
    """Analyze the characteristics of each cluster"""
    # Get numeric columns to analyze
    numeric_cols = original_df.columns.tolist()
    
    # Calculate mean values for each metric by cluster
    cluster_means = clustered_df.groupby('Cluster')[numeric_cols].mean()
    
    # Create radar chart for each cluster
    n_clusters = clustered_df['Cluster'].nunique()
    
    # Select top metrics for radar chart
    # First, identify metrics with the most variance between clusters
    cluster_variance = cluster_means.var()
    top_metrics = cluster_variance.sort_values(ascending=False).index[:min(8, len(cluster_variance))]
    
    # Create radar chart data
    cluster_radar_data = cluster_means[top_metrics]
    
    # Normalize data for radar chart
    scaler = StandardScaler()
    scaled_radar_data = pd.DataFrame(
        scaler.fit_transform(cluster_radar_data),
        index=cluster_radar_data.index,
        columns=cluster_radar_data.columns
    )
    
    # Plot radar chart
    angles = np.linspace(0, 2*np.pi, len(top_metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
    
    for cluster in range(n_clusters):
        values = scaled_radar_data.loc[cluster].tolist()
        values += values[:1]  # Complete the circle
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f'Cluster {cluster}')
        ax.fill(angles, values, alpha=0.1)
    
    # Set labels and title
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(top_metrics, size=10)
    plt.title('Cluster Characteristics (Normalized Scores)', size=15)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig('analysis_output/cluster_radar_chart.png', dpi=300)
    plt.close()
    
    # Create a summary table of cluster characteristics
    cluster_summary = pd.DataFrame()
    
    for cluster in range(n_clusters):
        models_in_cluster = clustered_df[clustered_df['Cluster'] == cluster].index.tolist()
        
        # Find metrics where this cluster stands out
        relative_performance = {}
        for metric in numeric_cols:
            if metric in cluster_means.columns:
                # Calculate how much this cluster deviates from the overall mean
                cluster_value = cluster_means[metric].loc[cluster]
                overall_mean = cluster_means[metric].mean()
                overall_std = cluster_means[metric].std()
                
                if overall_std > 0:  # Avoid division by zero
                    z_score = (cluster_value - overall_mean) / overall_std
                    relative_performance[metric] = (z_score, cluster_value)
        
        # Get top 3 distinctive metrics (highest absolute z-scores)
        top_metrics = sorted(relative_performance.items(), key=lambda x: abs(x[1][0]), reverse=True)[:3]
        top_metrics_str = ', '.join([f"{m} ({v[1]:.2f})" for m, v in top_metrics])
        
        # Add to summary
        cluster_summary[f'Cluster {cluster}'] = pd.Series({
            'Number of Models': len(models_in_cluster),
            'Models': ', '.join(models_in_cluster),
            'Distinctive Metrics': top_metrics_str
        })
    
    return cluster_summary, cluster_means



def debug_value(df, stage):
    """
    Print the value of "openbench_AIME 2024 (Pass@1)" for model "o1-2024-12-17-high" at a given stage.
    """
    # Depending on the stage, model names may be in a column or index.
    # First, try if model names are in a column called 'model_name'
    if 'model_name' in df.columns:
        debug_df = df[df['model_name'] == "o1-2024-12-17-high"]
    else:
        # If not, assume model names are the index
        debug_df = df.loc[["o1-2024-12-17-high"]] if "o1-2024-12-17-high" in df.index else None

    if debug_df is not None and not debug_df.empty:
        if "openbench_AIME 2024 (Pass@1)" in debug_df.columns:
            value = debug_df["openbench_AIME 2024 (Pass@1)"].values[0]
            print(f"{stage} - 'openbench_AIME 2024 (Pass@1)' for model 'o1-2024-12-17-high': {value}")
        else:
            print(f"{stage} - Column 'openbench_AIME 2024 (Pass@1)' not found.")
    else:
        print(f"{stage} - Model 'o1-2024-12-17-high' not found in DataFrame.")


# ==============================================================================
# MAIN ENTRY POINT
# ==============================================================================

def main(file_path):
    """Run the complete exploratory data analysis pipeline.

    Executes the full EDA workflow:
    1. Load and clean data
    2. Prepare for analysis (filter models/columns, handle missing values)
    3. Compute correlation matrix and identify redundant benchmarks
    4. Perform PCA for dimensionality understanding
    5. Perform t-SNE and optional UMAP for visualization
    6. Cluster models and analyze cluster profiles
    7. Generate visualizations and summary reports

    Args:
        file_path: Path to combined benchmark CSV file.

    Returns:
        Dict with analysis results including:
        - df: Cleaned data
        - analysis_df: Analysis-ready DataFrame
        - correlation_matrix: Pairwise correlations
        - pca_results: PCA coordinates and loadings
        - cluster_labels: Model-to-cluster assignments

    Outputs:
        Creates 'analysis_output/' directory with:
        - correlation_matrix.csv, correlation_heatmap.png
        - pca_explained_variance.png, pca_loadings.csv
        - tsne_visualization.png, umap_visualization.png (if UMAP available)
        - cluster_profiles.csv, cluster_radar_chart.png
        - Various scatter plots colored by company/cluster
    """
    print(f"Reading data from {file_path}...")
    
    # Read and clean data
    df = read_and_clean_data(file_path)
    print(f"Found {len(df)} models in the dataset.")
    
    if len(df) == 0:
        print("No valid data found. Exiting.")
        return None
    

    #for exclusion
    organized_columns = {
        'contextarena': [
            'contextarena_16k (%)',
            'contextarena_32k (%)',
            'contextarena_64k (%)',
            'contextarena_256k (%)',
            'contextarena_512k (%)',
            'contextarena_1M (%)',
            'contextarena_AUC @128k (%)',
            'contextarena_AUC @1M (%)',
            'contextarena_Model AUC (%)',
            'contextarena_Runs',
            'contextarena_Model_Mapped_to_OpenBench',
            'contextarena_Max Ctx',
            'contextarena_AUC @1M (%)',
            'contextarena_Model',
            'contextarena_Input Cost ($)',
            'contextarena_Output Cost ($)',
            'contextarena_128k (%)',
            'contextarena_Tok Eff.',
            'contextarena_8k (%) 8 needles',
            'contextarena_8k (%) 2 needles'
        ],
        'hallucinationbench': [
            'hallucinationbench_Average Summary Length (Words)',
            'hallucinationbench_Factual Consistency Rate',
            'hallucinationbench_Answer Rate',
            'hallucinationbench_Model_Mapped_to_OpenBench',
            'hallucinationbench_Model'
        ],
        'lechmazur': [
            'lechmazur_step_exposed',
            'lechmazur_Weighted',
            'lechmazur_elim_Exposed',
            'lechmazur_elim_Ratio',
            'lechmazur_step_expose',
            'lechmazur_step_ratio',
            'lechmazur_elim_Games',
            'lechmazur_step_p-wins',
            'lechmazur_confab_Weighted',
            'lechmazur_step_sigma',
            'lechmazur_step_games',
            'lechmazur_elim_Points',
            'lechmazur_elim_σ',
            'lechmazur_writing_Rank',
            'lechmazur_nytcon_#Puzzles',
            'lechmazur_elim_&sigma;',
            'lechmazur_elim_Exposed (&mu;)',
            'lechmazur_elim_Points Sum',
            'lechmazur_elim_Avg Points',
            'lechmazur_Model_Mapped_to_OpenBench',
            'lechmazur_elim_&mu;',
            'lechmazur_step_mu',
            'lechmazur_writing_Mean',
            'lechmazur_confab_Non-Resp %'
        ],
        'livebench': [
            'livebench_Global Average',
            'livebench_Coding Average',
            'livebench_Mathematics Average',
            'livebench_Data Analysis Average',
            'livebench_Language Average',
            'livebench_Reasoning Average',
            # 'livebench_IF Average',
            'livebench_paraphrase',
            'livebench_simplify',
            'livebench_story_generation',
            'livebench_summarize',
            'livebench_Unnamed: 0',
            'livebench_Agentic Coding Average',
            'livebench_web_of_lies_v3',
            'livebench_math_comp',
            # 'livebench_zebra_puzzle',
            # 'livebench_spatial',
            'livebench_Coding Average_dup',
            'livebench_Agentic Coding Average_dup',
            'livebench_Data Analysis Average_dup',
            'livebench_Language Average_dup',
            'livebench_IF Average_dup',
            'livebench_Model_Mapped_to_OpenBench',
            'livebench_AMPS_Hard',        # near-solved: 90% of models score 84+, ceiling effect (skew=-2.97)
            'livebench_tablereformat',    # near-solved: 50% score 98+, SD=3.3, best corr only 0.45
        ],
        'lmsys': [
            'lmsys_rank',
            'lmsys_rank_stylectrl',
            'lmsys_95_pct_ci',
            'lmsys_95% CI (±)',
            'lmsys_votes',
            'lmsys_organization',
            'lmsys_license',
            'lmsys_Votes',
            'lmsys_Organization',
            'lmsys_License',
            'lmsys_Model_Mapped_to_OpenBench'
        ],
        'lmarena': [
            'lmarena_Model',
            'lmarena_95% CI (±)', 
            'lmarena_Votes',
            'lmarena_Organization', 
            'lmarena_License',
            'lmarena_Model_Mapped_to_OpenBench',
            # 'lmarena_Score'
        ],
        'openbench': [
            'openbench_Input Cost Per Million Tokens ($)',
            "openbench_EtE Response Time 500 Output Tokens",
            'openbench_Total Parameters (Billion)',
            'openbench_Parameters (Billion)'
            'openbench_Total Parameters (Billion)',
            'openbench_Activated Parameters (Billion)',
            'openbench_Context Window (k)',
            'openbench_Organization',
            'openbench_Model_codename',
            'openbench_openbench_id',
            'openbench_AA-Omniscience Accuracy',
            'openbench_AA-Omniscience Hallucination Rate',
            'openbench_AA-Adj-Hallucination'

        ],
        'simplebench': [
            'simplebench_Model_Mapped_to_OpenBench'
        ],
        'logic': [
            'logic_correct',
            'logic_partially_correct',
            'logic_incorrect',
            'logic_accuracy_percent',
            'logic_other_judgments',
            'logic_model_id',
            'logic_model_name',
            'logic_total_questions',
            'logic_avg_output_tokens',
            'logic_total',
            'logic_no_answer',
            'logic_refusal',
            'logic_unsafe',
            'logic_label_skipped: model failed'

        ],
        'style': [
            'style_style_score',
            'style_model',
            'style_combined_length',
            'style_combined_header_count',
            'style_combined_bold_count',
            'style_combined_bold_count',
            'style_combined_list_count',
            'style_combined_list_count',
            'style_lmarena_Score',
            'style_lmsys_Score',
            'style_delta_score'
        ],
        'aiderbench': [
            'aiderbench_\n\n',
            'aiderbench_Edit Format',
            'aiderbench_Model_Mapped_to_OpenBench',
            'aiderbench_Model',
            'aiderbench_Correct edit format',
            'aiderbench_Cost'
        ],
        'eqbench': [
            'eqbench_Unnamed: 2',
            'eqbench_Model',
            'eqbench_Model_Mapped_to_OpenBench',
            "eqbench_model",
            "eqbench_eq_elo_ci_low",
            "eqbench_eq_elo_ci_high",
            "eqbench_Model_Mapped_to_OpenBench",
            "eqbench_eq_insightful",
            "eqbench_eq_pragmatic",
            "eqbench_eq_empathy",
            "eqbench_eq_humanlike",
            "eqbench_eq_rubric_score",
            "eqbench_eq_warm",
            "eqbench_eq_moral",
            "eqbench_eq_compliant",
            "eqbench_eq_analytical",
            "eqbench_eq_social_iq",
            "eqbench_eq_assertive",
            "eqbench_eq_safe",
            "eqbench_creative_rubric_score",
            "eqbench_creative_repetition_score",
            "eqbench_creative_length",
            "eqbench_creative_slop_score",
            "eqbench_creative_vocab_complexity"


        ] ,
        'tone' : [
            'tone_judged_model',
            'tone_Grok 4.1 Fast density Sigma',
            'tone_Grok 4.1 Fast confidence Sigma'
        ],
        'writing' : [
            'writing_writer_model',
            "writing_Qwen 3 32B_score",
            "writing_Qwen 3 235B A22B_score",
            'writing_Grok 4 Fast RD',
            "writing_Gemini 3.0 Flash Preview (2025-12-17)  RD",
            'writing_Gemini 3.0 Flash Preview (2025-12-17)  Glicko',
            'writing_Gemini 3.0 Flash Preview (2025-12-17)  TrueSkill',
            'writing_Gemini 3.0 Flash Preview (2025-12-17)  Sigma',
            'writing_Grok 4 Fast Sigma'

        ],
        'arc' : [
            'arc_Organization',
            'arc_Model',
            'arc_System Type',
            'arc_Cost/Task',
            'arc_Code / Paper',
            'arc_Model_Mapped_to_OpenBench',
            'arc_Author'
        ],
        'aa' : [
            'aa_id',
            'aa_Model',
            # 'aa_name',
            'aa_pricing_price_1m_blended_3_to_1',
            'aa_eval_artificial_analysis_coding_index',
            'aa_eval_artificial_analysis_intelligence_index',
            'aa_eval_artificial_analysis_math_index',
            "aa_Model_Mapped_to_OpenBench",
            'aa_eval_math_500',
            'aa_eval_aime'
            
        ],
        'aa' : [
            'aa_id',
            'aa_Model',
            # 'aa_name',
            'aa_pricing_price_1m_blended_3_to_1',
            'aa_eval_artificial_analysis_coding_index',
            'aa_eval_artificial_analysis_intelligence_index',
            'aa_eval_artificial_analysis_math_index',
            "aa_Model_Mapped_to_OpenBench",
            'aa_eval_math_500',
            'aa_eval_aime'
            
        ],
        'ugileaderboard' : [
            'ugileaderboard_Model'
        ],


        'weirdml' : [
            'weirdml_Model',
            'weirdml_internal_model_name',
            'weirdml_model_slug',
            'weirdml_avg_acc_standard_error',
            'weirdml_cost_per_run_usd',
            'weirdml_mean_total_output_tokens',
            'weirdml_code_len_p10',
            'weirdml_code_len_p90',
            'weirdml_exec_time_median_s',
            'weirdml_release_date',
            'weirdml_API source',
            'weirdml_Model_Mapped_to_OpenBench',
            'weirdml_shapes_easy_acc',
            'weirdml_digits_unsup_acc',
            'weirdml_chess_winners_acc',
            'weirdml_kolmo_shuffle_acc',
            'weirdml_classify_sentences_acc',
            'weirdml_insert_patches_acc',
            'weirdml_blunders_easy_acc',
            'weirdml_blunders_hard_acc',
            'weirdml_digits_generalize_acc',
            'weirdml_shapes_variable_acc',
            'weirdml_xor_easy_acc',
            'weirdml_xor_hard_acc',
            'weirdml_splash_easy_acc',
            'weirdml_splash_hard_acc',
            'weirdml_code_len_p50',
            'weirdml_number_patterns_acc',
            'weirdml_shapes_hard_acc',
            'weirdml_classify_shuffled_acc'
        ],

        'yupp' : [
            'yupp_Model_Mapped_to_OpenBench',
            'yupp_Model'
        ],

        'eq': [
            'eq_Grok 4 Fast Sigma',
            'eq_Gemini 3.0 Flash Preview (2025-12-17)  Sigma',
            'eq_Claude Opus 4.6 Thinking TrueSkill',
            'eq_Claude Opus 4.6 Thinking Sigma',
            'eq_model'
        ],

        'aagdpval': [
            'aagdpval_Model',
            'aagdpval_Creator',
            'aagdpval_CI_Lower',
            'aagdpval_CI_Upper',
            'aagdpval_Rank',
            'aagdpval_isReasoning',
            'aagdpval_Model_Mapped_to_OpenBench',
        ],

        'aaomniscience': [
            'aaomniscience_Model',
            'aaomniscience_Creator',
            'aaomniscience_isReasoning',
            'aaomniscience_Model_Mapped_to_OpenBench',
            'aaomniscience_OmniscienceHallucinationRate',
        ],

        'aacritpt': [
            'aacritpt_Model',
            'aacritpt_Creator',
            'aacritpt_Rank',
            'aacritpt_isReasoning',
            'aacritpt_Model_Mapped_to_OpenBench',
        ]

    }

    exclude_columns = []
    for key in organized_columns:
        exclude_columns.extend(organized_columns[key])

    analysis_df, analysis_df_non_imputed = prepare_data_for_analysis(
        df,
        exclude_columns=exclude_columns,
        include_missingness_flags=INCLUDE_MISSINGNESS_FLAGS
    )
    missing_flag_cols = [c for c in analysis_df.columns if c.endswith("__was_missing")]
    analysis_df_for_analysis = drop_missing_flag_columns(analysis_df)
    analysis_df_non_imputed_for_analysis = drop_missing_flag_columns(analysis_df_non_imputed)
    print(f"Selected {len(analysis_df.columns)} features total ({len(missing_flag_cols)} missingness flags); using {len(analysis_df_for_analysis.columns)} for correlations/PCA/clustering.")
    
    if len(analysis_df_for_analysis.columns) == 0:
        print("No numerical features found for analysis. Exiting.")
        return None
    
    # Analyze correlations
    print("Analyzing correlations between metrics...")
    corr_matrix, high_correlations = analyze_correlations(analysis_df_non_imputed_for_analysis)
    print(f"Found {len(high_correlations)} highly correlated feature pairs (|r| > 0.95).")
    
    # Print top correlations
    print("\nTop Correlated Metric Pairs:")
    # Iterate over the rows of the DataFrame returned by analyze_correlations
    for index, row in high_correlations.iterrows():
        feat1 = row['Feature 1']
        feat2 = row['Feature 2']
        corr = row['Correlation']
        # 'index + 1' gives us a 1-based count for the list
        print(f"{index + 1}. {feat1} & {feat2}: r = {corr:.3f}")
    
    # Perform PCA
    print("\nPerforming PCA...")
    pca_df, explained_variance, feature_importance = perform_pca(analysis_df_for_analysis)
    print(f"Explained variance ratios: {', '.join([f'{v:.2%}' for v in explained_variance])}")
    
    print("\nTop features in PC1:")
    pc1_features = feature_importance['PC1']
    top_pc1 = pc1_features.reindex(pc1_features.abs().sort_values(ascending=False).index[:10])
    for feat, value in top_pc1.items():
        print(f"  {feat}: {value:.3f}")

    print("\nTop features in PC2:")
    pc2_features = feature_importance['PC2']
    top_pc2 = pc2_features.reindex(pc2_features.abs().sort_values(ascending=False).index[:10])
    for feat, value in top_pc2.items():
        print(f"  {feat}: {value:.3f}")

    if 'PC3' in feature_importance.columns:
        print("\nTop features in PC3:")
        pc3_features = feature_importance['PC3']
        top_pc3 = pc3_features.reindex(pc3_features.abs().sort_values(ascending=False).index[:10])
        for feat, value in top_pc3.items():
            print(f"  {feat}: {value:.3f}")


    
    # Perform t-SNE
    print("\nPerforming t-SNE...")
    tsne_df = perform_tsne(analysis_df_for_analysis)
    
    # Perform UMAP if available
    umap_df = perform_umap(analysis_df_for_analysis) if USE_UMAP else None
    
    # Perform clustering
    print("\nPerforming clustering analysis...")
    clustered_df, n_clusters = perform_clustering(analysis_df_for_analysis, n_clusters_range=range(2,8))
    print(f"Optimal number of clusters: {n_clusters}")
    sorted_df = pca_df.sort_values(by="model_name")
    print(sorted_df)
    # Save clean combined CSV before visualization (which may fail on some setups)
    analysis_df_non_imputed.reset_index().to_csv("benchmarks/clean_combined_all_benches.csv", index=False)
    print("Saved benchmarks/clean_combined_all_benches.csv")

    # Visualize reduced dimensions
    print("\nCreating visualizations...")
    visualize_reduced_dimensions(pca_df, tsne_df, umap_df, clustered_df, original_df=df)
    # visualize_reduced_dimensions(pca_df, tsne_df, umap_df, clustered_df)
    
    # Analyze clusters
    print("\nAnalyzing cluster characteristics...")
    cluster_summary, cluster_means = analyze_clusters(clustered_df, analysis_df_for_analysis)
    
    # Display cluster summary
    print("\nCluster Summary:")
    for col in cluster_summary.columns:
        print(f"\n{col}:")
        for idx, val in cluster_summary[col].items():
            print(f"  {idx}: {val}")
    
    print("\nAnalysis complete! Visualizations saved to current directory.")
    
    return {
        'original_data': df,
        'analysis_data': analysis_df,
        'analysis_data_for_analysis': analysis_df_for_analysis,
        'analysis_df_non_imputed': analysis_df_non_imputed,
        'analysis_df_non_imputed_for_analysis': analysis_df_non_imputed_for_analysis,
        'pca_results': pca_df,
        'tsne_results': tsne_df,
        'umap_results': umap_df,
        'clustered_data': clustered_df,
        'correlation_matrix': corr_matrix,
        'feature_importance': feature_importance,
        'cluster_summary': cluster_summary
    }

# If the script is run directly
if __name__ == "__main__":
    import sys
    
    # Use command line argument for file path if provided, otherwise use default
    file_path = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/combined_all_benches.csv"
    
    # Run the analysis
    results = main(file_path)


    results['analysis_df_non_imputed'].reset_index().to_csv("benchmarks/clean_combined_all_benches.csv", index=False)
