import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def prepare_data(df, columns_to_scale=None):
    """
    Isolates 'Player_ID', drops it from training data, and scales the specified columns.
    If columns_to_scale is None, scales all numeric columns except 'Player_ID'.
    """
    if 'Player_ID' in df.columns:
        train_df = df.drop(columns=['Player_ID'])
        player_ids = df['Player_ID']
    else:
        train_df = df.copy()
        player_ids = None
        
    if columns_to_scale is None:
        columns_to_scale = train_df.select_dtypes(include=['number']).columns.tolist()
        
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(train_df[columns_to_scale])
    
    # Create a DataFrame with the scaled data to maintain column names
    scaled_df = pd.DataFrame(scaled_data, columns=columns_to_scale, index=train_df.index)
    
    # If there were other columns not scaled, we might want to include them, 
    # but for K-Means we typically only use the scaled numerical columns.
    
    return scaled_df, player_ids, scaler

def calculate_elbow(scaled_df, max_k=8):
    """
    Calculates inertia for K=1 to max_k.
    Returns a list of inertias.
    """
    inertias = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(scaled_df)
        inertias.append(kmeans.inertia_)
    return inertias

def train_kmeans(scaled_df, n_clusters):
    """
    Trains the final K-Means model with optimal K.
    Returns the trained model, cluster labels, and centroids.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(scaled_df)
    centroids = kmeans.cluster_centers_
    return kmeans, labels, centroids
