import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import get_esports_data
from ml_pipeline import prepare_data, calculate_elbow, train_kmeans

st.set_page_config(page_title="Esports Player Analyst", layout="wide")

st.title("The Agentic Esports Data Analyst")

# Load Data
df = get_esports_data()

st.subheader("Raw Dataset")
st.dataframe(df)

# We use all numeric columns for a better clustering, or specifically survival and damage.
# Let's use all features for a complete profile.
columns_to_scale = ['Average_Survival_Time_Min', 'Average_Damage_Dealt', 'Kills_Per_Match', 'Assists_Per_Match']

scaled_df, player_ids, scaler = prepare_data(df, columns_to_scale)

# Elbow Method
st.subheader("Determining Optimal K with the Elbow Method")
inertias = calculate_elbow(scaled_df, max_k=8)

fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
ax_elbow.plot(range(1, 9), inertias, marker='o', linestyle='--')
ax_elbow.set_title("Elbow Method for Optimal K")
ax_elbow.set_xlabel("Number of Clusters (K)")
ax_elbow.set_ylabel("Inertia")
ax_elbow.grid(True)
st.pyplot(fig_elbow)

# Optimal K is likely 5 based on the generated data
optimal_k = st.slider("Select Optimal Number of Clusters (K)", min_value=2, max_value=8, value=5)

# Train K-Means
kmeans_model, labels, centroids = train_kmeans(scaled_df, optimal_k)

# Add cluster labels back to the original dataframe for visualization
df_clustered = df.copy()
df_clustered['Cluster'] = labels

# Scatter Plot
st.subheader(f"Player Clusters (K={optimal_k})")
st.write("Visualizing clusters based on Survival Time and Damage Dealt.")

fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))

# Plot the points
sns.scatterplot(
    data=df_clustered, 
    x='Average_Survival_Time_Min', 
    y='Average_Damage_Dealt', 
    hue='Cluster', 
    palette='Set1', 
    s=100, 
    alpha=0.8, 
    ax=ax_scatter
)

# Unscale centroids to plot them on the original scale
# Since we scaled 4 features, we need to unscale them. Centroids have 4 dimensions.
# We will unscale all, and then select the first two columns corresponding to Survival and Damage
unscaled_centroids = scaler.inverse_transform(centroids)
centroid_survival = unscaled_centroids[:, 0]
centroid_damage = unscaled_centroids[:, 1]

# Plot the centroids
ax_scatter.scatter(
    centroid_survival, 
    centroid_damage, 
    s=300, 
    c='black', 
    marker='*', 
    label='Centroids'
)

ax_scatter.set_title("Esports Player Clusters: Survival vs Damage")
ax_scatter.set_xlabel("Average Survival Time (Minutes)")
ax_scatter.set_ylabel("Average Damage Dealt")
ax_scatter.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
st.pyplot(fig_scatter)

st.subheader("Clustered Data")
st.dataframe(df_clustered)
