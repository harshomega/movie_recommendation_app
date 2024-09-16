# Install dependencies
# pip install torch scikit-learn streamlit

# Import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import streamlit as st

# Load MovieLens dataset
def load_data():
    # Download MovieLens dataset from https://grouplens.org/datasets/movielens/
    ratings = pd.read_csv("C:\\Users\msi 1\Desktop\movies_ratings.csv")  # Use the appropriate path for your MovieLens data
    return ratings

ratings = load_data()

# Preprocess data
user_ids = ratings["userId"].unique().tolist()
movie_ids = ratings["movieId"].unique().tolist()

# Create a mapping for users and movies
user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
movie_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}

ratings['userId'] = ratings['userId'].map(user_map)
ratings['movieId'] = ratings['movieId'].map(movie_map)

# Create training and test datasets
train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)

# PyTorch Matrix Factorization Model
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, embedding_size=50):
        super(MatrixFactorization, self).__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_size)
        self.item_embedding = nn.Embedding(num_items, embedding_size)
    
    def forward(self, user, item):
        user_embedding = self.user_embedding(user)
        item_embedding = self.item_embedding(item)
        return (user_embedding * item_embedding).sum(1)

# Hyperparameters
num_users = ratings['userId'].nunique()
num_movies = ratings['movieId'].nunique()
embedding_size = 50
epochs = 20
learning_rate = 0.01

# Create model
model = MatrixFactorization(num_users, num_movies, embedding_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

# Convert data to PyTorch tensors
train_users = torch.tensor(train_data['userId'].values, dtype=torch.long)
train_movies = torch.tensor(train_data['movieId'].values, dtype=torch.long)
train_ratings = torch.tensor(train_data['rating'].values, dtype=torch.float32)

test_users = torch.tensor(test_data['userId'].values, dtype=torch.long)
test_movies = torch.tensor(test_data['movieId'].values, dtype=torch.long)
test_ratings = torch.tensor(test_data['rating'].values, dtype=torch.float32)

# Train the model
for epoch in range(epochs):
    model.train()
    
    optimizer.zero_grad()
    output = model(train_users, train_movies)
    loss = loss_fn(output, train_ratings)
    loss.backward()
    optimizer.step()
    
    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# Evaluate the model
model.eval()
with torch.no_grad():
    predictions = model(test_users, test_movies)
    test_loss = mean_squared_error(test_ratings.numpy(), predictions.numpy())
    print(f'Test MSE: {test_loss}')

# Streamlit deployment for recommendation
def recommend_movies(user_id, num_recommendations=5):
    # Recommend top movies for a given user
    model.eval()
    user_idx = user_map.get(user_id)
    if user_idx is None:
        return "User not found."
    
    user_tensor = torch.tensor([user_idx] * num_movies, dtype=torch.long)
    movie_tensor = torch.tensor(list(range(num_movies)), dtype=torch.long)
    
    with torch.no_grad():
        scores = model(user_tensor, movie_tensor).numpy()
    
    recommended_movie_indices = np.argsort(-scores)[:num_recommendations]
    recommended_movie_ids = [movie_ids[idx] for idx in recommended_movie_indices]
    
    return recommended_movie_ids

# Streamlit UI
st.title("Movie Recommendation System")

user_input = st.text_input("Enter User ID for Recommendations:", "1")
if st.button("Recommend Movies"):
    recommendations = recommend_movies(int(user_input), num_recommendations=5)
    st.write(f"Recommended Movies for User {user_input}:")
    for movie in recommendations:
        st.write(f"Movie ID: {movie}")
