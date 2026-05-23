import plotly.express as px
import plotly.graph_objects as go
from dataset import get_esports_data
from ml_pipeline import prepare_data, train_kmeans

def create_dashboard():
    # 1. Load Data
    df = get_esports_data()
    
    # 2. Scale Data and isolate Player_ID
    # We want to scale all numeric columns or specifically the 4 core ones
    columns_to_scale = ['Average_Survival_Time_Min', 'Average_Damage_Dealt', 'Kills_Per_Match', 'Assists_Per_Match']
    scaled_df, player_ids, scaler = prepare_data(df, columns_to_scale)
    
    # 3. Train K-Means (we use K=5 based on previous elbow analysis)
    optimal_k = 5
    kmeans_model, labels, centroids = train_kmeans(scaled_df, optimal_k)
    
    # 4. Attach labels back to original df for visualization
    df['Cluster'] = labels
    
    # Map clusters to descriptive MOBA roles dynamically based on stats
    cluster_names = {}
    for i in range(optimal_k):
        c_df = df[df['Cluster'] == i]
        surv = c_df['Average_Survival_Time_Min'].mean()
        dmg = c_df['Average_Damage_Dealt'].mean()
        assists = c_df['Assists_Per_Match'].mean()
        
        if surv < 10 and dmg > 1200:
            name = "Assassin"
        elif surv > 20 and dmg < 450:
            name = "Tank"
        elif surv > 20 and assists > 5:
            name = "Support"
        elif surv < 10 and dmg < 400:
            name = "Mage"
        else:
            name = "Fighter"
        cluster_names[i] = name
        
    df['Role'] = df['Cluster'].map(cluster_names)
    
    # 5. Create Plotly Express Scatter Plot
    fig = px.scatter(
        df,
        x='Average_Survival_Time_Min',
        y='Average_Damage_Dealt',
        color='Role',
        hover_data=['Player_ID', 'Kills_Per_Match', 'Assists_Per_Match', 'Role'],
        title="Esports Player Roles (Fighter, Assassin, Mage, Tank, Support)",
        labels={
            'Average_Survival_Time_Min': 'Average Survival Time (Minutes)',
            'Average_Damage_Dealt': 'Average Damage Dealt'
        },
        template='plotly_dark'
    )
    
    # 6. Unscale Centroids to map them back to original coordinates
    unscaled_centroids = scaler.inverse_transform(centroids)
    # The columns_to_scale were: Survival(0), Damage(1), Kills(2), Assists(3)
    centroid_survival = unscaled_centroids[:, 0]
    centroid_damage = unscaled_centroids[:, 1]
    
    # 7. Add Centroids as large red stars
    fig.add_trace(
        go.Scatter(
            x=centroid_survival,
            y=centroid_damage,
            mode='markers',
            marker=dict(symbol='star', size=20, color='red', line=dict(width=2, color='DarkSlateGrey')),
            name='Centroids',
            hoverinfo='skip'
        )
    )
    
    # 8. Export to HTML
    fig.write_html("dashboard.html")
    print("Successfully generated dashboard.html")

if __name__ == "__main__":
    create_dashboard()
