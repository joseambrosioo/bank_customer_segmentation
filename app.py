import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import dash_table # Import dash_table for displaying data

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
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], title="Bank Customer Segmentation")
server = app.server

# Custom color palette for the clusters
cluster_colors = px.colors.qualitative.Plotly[:7]

# Header for the dashboard, mimicking the new style
header = dbc.Navbar(
    dbc.Container([
        html.Div([
            html.Span("💳", className="me-2"),
            dbc.NavbarBrand("Bank Customer Segmentation", class_name="fw-bold text-wrap", style={"color": "black"}),
        ], className="d-flex align-items-center"),
        dbc.Badge("Dashboard", color="primary", className="ms-auto")
    ]),
    color="light",
    class_name="shadow-sm mb-3"
)

# --- Tab Content ---
ask_tab_content = dcc.Markdown(
    """
### ❓ **ASK** — The Business Question
This project is designed to help a bank's marketing team move beyond a one-size-fits-all approach. Instead, we're using **customer segmentation** to group customers with similar spending behaviors. This allows the marketing team to create highly targeted ad campaigns that are more likely to succeed.

**Stakeholders**: The primary audience for this dashboard is the **Marketing Team** and **Bank Management**. The marketing team will use these insights to design specific campaigns, while management can track customer behavior and the success of our segmentation strategy.

**Deliverables**: This interactive dashboard serves as the final deliverable, providing a clear, visual, and explained overview of our analysis, from raw data to actionable customer groups.
    """,
    className="p-4"
)

prepare_tab_content = html.Div([
    dcc.Markdown("### 📝 **PREPARE** — Getting the Data Ready", className="p-2"),
    html.P("This section covers the data we used and the key steps taken to prepare it for analysis."),
    
    html.H5("Data Source"),
    html.P("We analyzed a dataset containing 8,950 records of credit card usage data for the past six months. Each record represents a single customer's aggregated behavior, with features like card balance, purchase types, payment history, and card tenure."),

    html.H5("Data Cleaning and Preprocessing"),
    html.P("Before we could apply any machine learning algorithms, the data had to be cleaned and transformed. Here's what we did:"),
    html.Ul([
        html.Li(dcc.Markdown("**Handling Missing Values:** We found some missing data points in the 'MINIMUM_PAYMENTS' (313 rows) and 'CREDIT_LIMIT' (1 row) columns. To ensure our analysis was accurate and complete, we filled these gaps with the average value of each respective column.")),
        html.Li(dcc.Markdown("**Removing Irrelevant Features:** The 'CUST_ID' column was removed because it's a unique identifier for each customer and doesn't provide any behavioral information that would be useful for grouping them.")),
        html.Li(dcc.Markdown("**Data Scaling:** Clustering algorithms like K-Means are very sensitive to the scale of the data. For example, a feature like 'PURCHASES' might have values in the thousands, while 'TENURE' (the number of months a card has been active) is much smaller. To prevent features with larger values from unfairly dominating the results, we used a **StandardScaler** to normalize all features so they contribute equally to the distance calculations.")),
    ]),
    
    html.H5("Data Summary"),
    html.P("Here's a quick look at the statistical summary of the prepared data."),
    dbc.Table.from_dataframe(creditcard_df.describe().round(2).T, striped=True, bordered=True, hover=True),

    html.H5("Dataset Sample (First 10 Rows)"),
    dash_table.DataTable(
        id='table',
        columns=[{"name": i, "id": i} for i in creditcard_df.columns],
        data=creditcard_df.head(10).to_dict('records'),
        sort_action="native",  # Enable sorting
        filter_action="native", # Enable filtering
        page_action="native", # Corrected from "none" to "native"
        page_size=10, # Added to ensure 10 rows are displayed per page
        style_table={'overflowX': 'auto', 'width': '100%'},
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold',
            'textAlign': 'center',
        },
        style_cell={
            'textAlign': 'left',
            'padding': '5px',
            'font-size': '12px',
            'minWidth': '80px', 'width': 'auto', 'maxWidth': '150px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
        },
    ),
    html.Br(),
], className="p-4")


analyze_tab_content = html.Div([
    dcc.Markdown("### 📈 **ANALYZE** — Finding Patterns and Creating Segments", className="p-2"),
    html.P("This is where we apply the K-Means algorithm to find natural customer groups. We then use Principal Component Analysis (PCA) to visualize these groups in a simpler way."),
    
    html.Hr(),
    html.H5("Step 1: Finding the Optimal Number of Clusters (The Elbow Method)"),
    html.P(dcc.Markdown("K-Means requires us to specify the number of clusters (K) in advance. We used the **Elbow Method** to determine the optimal number. This method involves running the clustering for a range of `K` values and plotting the **Within-Cluster Sum of Squares (WCSS)**. WCSS measures how compact and dense the clusters are. A lower WCSS means a better clustering model. We look for the 'elbow' point on the graph—where the rate of decrease in WCSS slows down significantly—which indicates a good balance between a small number of clusters and a low WCSS. Our analysis shows a clear elbow at **K=7**.")),

    dbc.Card(
        dbc.CardBody(
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
            )
        )
    ),

    html.Hr(),
    html.H5("Step 2: Visualizing Customer Segments with PCA"),
    html.P(dcc.Markdown("Our dataset has 17 different features, which is too many to visualize in a single chart. To solve this, we used **Principal Component Analysis (PCA)**. PCA is a powerful technique that simplifies complex data by reducing the number of features while keeping the most important information. We reduced our data to just two main components (**PCA1** and **PCA2**) that capture the most variance, allowing us to plot all our customers and see how the different clusters are separated in a 2D space.")),

    dbc.Card(
        dbc.CardBody(
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
            )
        )
    ),
    
    html.Hr(),
    html.H5("Step 3: Cluster Profiling and Descriptions"),
    html.P("To understand the unique characteristics of each cluster, we can compare the average values of the original features. Use the dropdown below to explore how each cluster's behavior differs."),
    html.P(dcc.Markdown("This comparison helps us give a 'personality' to each cluster. For example, by selecting 'BALANCE', you can see which cluster tends to carry the highest average balance. This is different from a simple total because it tells you about the *behavior of the typical customer* in that group. The same applies to other features like `ONEOFF_PURCHASES` (single, large purchases) versus `INSTALLMENTS_PURCHASES` (many small, planned purchases). Separating these metrics gives us a much clearer picture of what a customer values.")),
    dbc.Row([
        dbc.Col([
            dbc.Label("Select a feature to compare clusters:"),
            dcc.Dropdown(
                id='feature-dropdown',
                options=[{'label': col, 'value': col} for col in creditcard_df.columns],
                value='BALANCE',
                clearable=False,
            )
        ], md=4),
        dbc.Col(
            dbc.Card(
                dbc.CardBody(
                    dcc.Graph(id='cluster-feature-bar-chart')
                )
            ), md=8
        )
    ], className="g-3"),
    
    # html.Hr(),
    html.H5("Detailed Cluster Descriptions:"),
    
    html.Ul([
        html.Li(dcc.Markdown("**Cluster 0: The High-Value Spenders** - These customers have a high credit limit, make frequent and large one-off purchases, and pay off their balances quickly. They represent the most profitable and engaged segment.")),
        html.Li(dcc.Markdown("**Cluster 1: The 'Revolvers' (Credit-as-a-Loan Users)** - This group carries the highest balances and cash advances. They use their credit card more as a source of short-term loans and are a high-risk group due to potential debt.")),
        html.Li(dcc.Markdown("**Cluster 2: The New Customers** - This cluster is defined by the lowest tenure, meaning they are new to the bank. They have low activity across all metrics, representing a crucial opportunity for loyalty-building efforts.")),
        html.Li(dcc.Markdown("**Cluster 3: The Installment Shoppers** - These customers make a high number of purchases and pay frequently, often in installments. They are careful with large one-off purchases and prefer to budget their spending.")),
        html.Li(dcc.Markdown("**Cluster 4: The Cash-Advance Users** - This segment is characterized by a high frequency of cash advance transactions, but relatively low purchase activity. They likely use their credit card for short-term liquidity, which is a high-cost habit.")),
        html.Li(dcc.Markdown("**Cluster 5: The Infrequent Users** - This group has a low balance and low activity across all transaction metrics. They may have a card but rarely use it, representing a low-cost, low-profit segment.")),
        html.Li(dcc.Markdown("**Cluster 6: The Budget-Minded Spenders** - This group makes a high number of purchases but with a low average transaction amount. They are disciplined with their payments and are not looking for large credit lines.")),
    ], className="mt-3")
], className="p-4")

# Update act_tab_content to be a pure summary of recommendations
act_tab_content = html.Div([
    dcc.Markdown("### 🚀 **ACT** — Turning Insights into Actionable Strategies", className="p-2"),
    html.P(dcc.Markdown("Based on our detailed analysis, we have identified and profiled 7 distinct customer segments. The marketing team can use the following high-level recommendations to design targeted campaigns that are far more effective than a generic, one-size-fits-all approach. This is the ultimate goal of the entire project.")),
    
    html.Hr(),
    html.H5("Marketing Recommendations by Segment:"),
    html.Ul([
        html.Li(dcc.Markdown("**Cluster 0 (High-Value Spenders):** *Action: Offer exclusive rewards programs for high-value purchases or increase their credit limit to encourage further spending.*")),
        html.Li(dcc.Markdown("**Cluster 1 ('Revolvers' (Credit-as-a-Loan Users)):** *Action: Promote balance transfer offers with lower interest rates or offer financial management tools to help them reduce debt and improve their financial health.*")),
        html.Li(dcc.Markdown("**Cluster 2 (New Customers):** *Action: Focus on onboarding programs, educational content about credit card benefits, and special offers to encourage initial engagement and loyalty.*")),
        html.Li(dcc.Markdown("**Cluster 3 (Installment Shoppers):** *Action: Partner with popular retailers to offer special installment plans and discounts to drive more frequent installment usage.*")),
        html.Li(dcc.Markdown("**Cluster 4 (Cash-Advance Users):** *Action: Promote balance transfer options and financial wellness tools to help them manage their high-interest debt effectively.*")),
        html.Li(dcc.Markdown("**Cluster 5 (Infrequent Users):** *Action: Drive engagement with incentives for small, frequent purchases (e.g., a bonus for using the card 3 times in a month). Remind them of the card's rewards and benefits.*")),
        html.Li(dcc.Markdown("**Cluster 6 (Budget-Minded Spenders):** *Action: Offer rewards for small, everyday transactions and highlight the convenience of using the card for daily purchases.*")),
    ], className="mt-3")
], className="p-4")

app.layout = dbc.Container([
    header,
    dbc.Tabs([
        dbc.Tab(ask_tab_content, label="Ask"),
        dbc.Tab(prepare_tab_content, label="Prepare"),
        dbc.Tab(analyze_tab_content, label="Analyze"),
        dbc.Tab(act_tab_content, label="Act"),
    ], className="mb-3", id="tabs"),
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