import pickle

# Assume user_map, movie_map, and model are already defined and trained
recommendation_data = {
    'user_map': user_map,
    'movie_map': movie_map,
    'model_state_dict': model.state_dict()  # Save the trained model's state
}

# Save as a pickle file
with open('recommendation_system_data.pkl', 'wb') as file:
    pickle.dump(recommendation_data, file)

print("Pickle file saved successfully!")

with open('recommendation_system_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)

# Access the mappings and model
user_map = loaded_data['user_map']
movie_map = loaded_data['movie_map']
model_state_dict = loaded_data['model_state_dict']

# Load the state_dict back into the model
model.load_state_dict(model_state_dict)
