import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# --- 1. Data Loading and Preprocessing ---
# This section handles all the data preparation, including loading, cleaning,
# and scaling, just as you did in your initial analysis.

creditcard_df = pd.read_csv("dataset/CC_GENERAL.csv") # Assuming the file is in the same directory

# Fill missing values with the mean
creditcard_df.loc[(creditcard_df['MINIMUM_PAYMENTS'].isnull() == True), 'MINIMUM_PAYMENTS'] = creditcard_df['MINIMUM_PAYMENTS'].mean()
creditcard_df.loc[(creditcard_df['CREDIT_LIMIT'].isnull() == True), 'CREDIT_LIMIT'] = creditcard_df['CREDIT_LIMIT'].mean()

# Drop the customer ID as it's not a useful feature for clustering
creditcard_df.drop('CUST_ID', axis=1, inplace=True)

# Scale the data for K-Means clustering
scaler = StandardScaler()
creditcard_df_scaled = scaler.fit_transform(creditcard_df)

# --- 2. K-Means Clustering ---
# Apply the K-Means algorithm to segment the customers. We'll use 7 clusters,
# based on your elbow method analysis.

# First, run the Elbow Method to determine the optimal number of clusters
wcss_scores = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(creditcard_df_scaled)
    wcss_scores.append(kmeans.inertia_)

# Based on the elbow plot, we choose 7 clusters for the final model
kmeans = KMeans(n_clusters=7, init='k-means++', max_iter=300, n_init=10, random_state=42)
labels = kmeans.fit_predict(creditcard_df_scaled)

# Add the cluster labels to the original dataframe for analysis
creditcard_df_cluster = pd.concat([creditcard_df, pd.DataFrame({'cluster': labels})], axis=1)

# --- 3. Principal Component Analysis (PCA) ---
# Reduce the dimensionality of the data to 2 components for visualization.

pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_df_scaled)

# Create a dataframe with the PCA components and the cluster labels
pca_df = pd.DataFrame(data=principal_comp, columns=['pca1', 'pca2'])
pca_df = pd.concat([pca_df, pd.DataFrame({'cluster': labels})], axis=1)

# --- 4. Dashboard App Layout ---

app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
server = app.server

# A custom color palette for the clusters
cluster_colors = px.colors.qualitative.Plotly[:7]

# Define the text content for each tab based on the ASK-PREPARE-ANALYZE-ACT framework
ask_tab_content = dcc.Markdown(
    """
    ### ❓ **ASK** — The Big Picture

    This project aims to help the bank's marketing team launch a more effective ad campaign. Instead of a one-size-fits-all approach, we will use **customer segmentation** to group customers with similar spending behaviors. This allows the marketing team to create tailored campaigns that resonate with each specific group, which is crucial for maximizing conversion rates and increasing profitability.

    **Stakeholders**: The primary users of this dashboard are the **Marketing Team** and **Bank Management**. The marketing team will use the insights to design targeted campaigns, while management can monitor customer behavior and the effectiveness of our segmentation strategy.

    **Deliverables**: This interactive dashboard serves as the final deliverable, providing a visual and explained overview of our analysis, from raw data to actionable customer groups.
    """
)

prepare_tab_content = html.Div([
    html.H4("📝 **PREPARE** — Getting the Data Ready"),
    html.P("This section covers the data we used and the steps taken to prepare it for analysis."),
    html.H5("Data Source"),
    html.P("We were provided with a dataset containing 8,950 rows and 18 columns of credit card usage data for the past six months. Each row represents a single customer's aggregated behavior, with features like balance, purchases, payments, and tenure."),
    html.H5("Data Cleaning and Preprocessing"),
    html.Ul([
        html.Li([html.B("Handling Missing Values:"), " We found some missing values in the 'MINIMUM_PAYMENTS' (313 rows) and 'CREDIT_LIMIT' (1 row) columns. We filled these gaps with the average value of each respective column to maintain data integrity without losing valuable information."]),
        html.Li([html.B("Removing Irrelevant Features:"), " The 'CUST_ID' column was removed as it is a unique identifier and holds no value for clustering customers based on their behavior."]),
        html.Li([html.B("Data Scaling:"), " Clustering algorithms like K-Means are very sensitive to the scale of the data. Features with larger values (e.g., PURCHASES, BALANCE) can disproportionately influence the results. To prevent this, we used a ", html.B("StandardScaler"), " to normalize all features, ensuring they contribute equally to the distance calculations."]),
    ]),
    html.H5("Data Insights"),
    html.P("Here's a quick look at the data's statistical summary after cleaning and before clustering."),
    dbc.Table.from_dataframe(creditcard_df.describe().round(2).T, striped=True, bordered=True, hover=True),
])

analyze_tab_content = html.Div([
    html.H4("📈 **ANALYZE** — Finding Patterns and Creating Segments"),
    html.P("This is where we apply the K-Means algorithm to find natural customer groups and then use Principal Component Analysis (PCA) to visualize these groups in a simpler way."),
    
    html.H5("Finding the Right Number of Clusters (The Elbow Method)"),
    html.P("K-Means requires us to specify the number of clusters (K) in advance. We used the **Elbow Method** to determine the optimal number. We plot the 'Within-Cluster Sum of Squares' (WCSS), which measures how compact the clusters are, against the number of clusters. The 'elbow' point on the graph—where the rate of decrease slows down—suggests a good balance. In our case, the elbow is around **7 clusters**."),
    
    dcc.Graph(
        id='elbow-plot',
        figure=go.Figure(
            data=[go.Scatter(x=list(range(1, 20)), y=wcss_scores, mode='lines+markers')],
            layout=go.Layout(
                title="Optimal Number of Clusters (K)",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Within-Cluster Sum of Squares (WCSS)"
            )
        )
    ),

    html.H5("Visualizing Customer Segments with PCA"),
    html.P("Our dataset has 17 features, making it impossible to visualize on a simple chart. **Principal Component Analysis (PCA)** is a technique that reduces the number of features while preserving the most important information. We reduced our data to two main components (PCA1 and PCA2) that capture the most variance, allowing us to plot all our customers and see how the clusters are separated."),

    dcc.Graph(
        id='pca-plot',
        figure=px.scatter(
            pca_df,
            x='pca1',
            y='pca2',
            color=pca_df['cluster'].astype(str),
            title='Customer Segmentation Clusters (PCA)',
            labels={'pca1': 'Principal Component 1', 'pca2': 'Principal Component 2', 'color': 'Cluster'},
            color_discrete_sequence=cluster_colors
        )
    ),
])

act_tab_content = html.Div([
    html.H4("🚀 **ACT** — Turning Insights into Actionable Strategies"),
    html.P("This is the most critical part: translating our data-driven clusters into real-world business strategies. Each of the 7 clusters represents a distinct customer group with unique behaviors that the marketing team can target."),
    html.P("We can describe the typical behavior of each cluster by looking at the average values of the original features within each group. The interactive bar chart below allows you to select a feature and see how the different clusters compare."),
    
    html.Div([
        dbc.Label("Select a feature to compare clusters:"),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[{'label': col, 'value': col} for col in creditcard_df.columns],
            value='BALANCE',
            clearable=False,
        ),
    ], className="mb-3"),
    dcc.Graph(id='cluster-feature-bar-chart'),

    html.H5("Example Segments and Marketing Recommendations:"),
    html.Ul([
        html.Li([
            html.B("Cluster 0: The Big Spenders"),
            " - These customers have a high credit limit, make frequent and large one-off purchases, and pay off their balances quickly. ",
            html.Em("Action: Offer exclusive rewards programs for high-value purchases or increase their credit limit to encourage further spending.")
        ]),
        html.Li([
            html.B("Cluster 1: The 'Revolvers' (Credit-as-a-Loan Users)"),
            " - This group carries the highest balances and cash advances. They tend to use their credit card more as a source of short-term loans. ",
            html.Em("Action: Promote balance transfer offers with lower interest rates or offer financial management tools to help them reduce debt.")
        ]),
        html.Li([
            html.B("Cluster 2: The New Customers"),
            " - This cluster is defined by the lowest tenure. They are new to the bank and have low activity across all metrics. ",
            html.Em("Action: Focus on onboarding programs, educational content about credit card benefits, and special offers to encourage initial engagement and loyalty.")
        ]),
        html.Li([
            html.B("Cluster 3: The Installment Shoppers"),
            " - These customers make a high number of installment purchases and pay frequently. They are careful with one-off purchases. ",
            html.Em("Action: Partner with popular retailers to offer special installment plans and discounts to drive more frequent installment usage.")
        ]),
    ], className="mt-3")
])


app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("💳 Bank Customer Segmentation", className="text-center my-4"))),
    
    dbc.Tabs([
        dbc.Tab(ask_tab_content, label="Ask"),
        dbc.Tab(prepare_tab_content, label="Prepare"),
        dbc.Tab(analyze_tab_content, label="Analyze"),
        dbc.Tab(act_tab_content, label="Act"),
    ]),
    
], fluid=True, className="p-4")


# --- Dashboard Callbacks ---
@app.callback(
    Output('cluster-feature-bar-chart', 'figure'),
    [Input('feature-dropdown', 'value')]
)
def update_bar_chart(selected_feature):
    # Calculate the average of the selected feature for each cluster
    cluster_means = creditcard_df_cluster.groupby('cluster')[selected_feature].mean().reset_index()
    
    fig = px.bar(
        cluster_means,
        x='cluster',
        y=selected_feature,
        title=f'Average {selected_feature} by Customer Cluster',
        labels={'cluster': 'Cluster', selected_feature: f'Average {selected_feature}'},
        color='cluster',
        color_discrete_sequence=cluster_colors,
    )
    fig.update_layout(xaxis={'categoryorder': 'total ascending'})
    return fig

if __name__ == '__main__':
    app.run(debug=True)