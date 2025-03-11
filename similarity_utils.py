from sklearn.metrics.pairwise import cosine_similarity


# Filter the dataset based on the cluster label
def filter_dataset(cluster_label, df):
    filtered_data = df[df["Cluster_Label"] == cluster_label]
    return filtered_data


# Function to calculate similarity based on user inputs
def calculate_similarity(user_inputs, filtered_data, n=5):
    # Extract features from user inputs
    user_features = [val for val in user_inputs]
    # Extract features from filtered dataset
    filtered_features = filtered_data[
        [
            key
            for key in [
                "country",
                "availability",
                "score",
                "flavor",
                "is_dark",
                "alcohol_lvl",
            ]
        ]
    ]

    # Calculate cosine similarity between user inputs and filtered dataset
    print(user_features)
    print(filtered_features)
    similarities = cosine_similarity([user_features], filtered_features)

    # Get indices of top similar rows
    top_indices = similarities.argsort(axis=1)[:, -n:][0][::-1]

    return top_indices


# Function to obtain top n rows with beer_name feature
def get_top_beers(top_indices, filtered_data, n):
    top_beers = filtered_data.iloc[top_indices[:n]]["beer_name"]
    return top_beers
