# Import necessary libraries
from typing import Tuple

import numpy as np  # To generate random data
import pandas as pd  # To handle data in tabular form
from sklearn.preprocessing import LabelEncoder


def get_dataset() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Step 1: Define the number of users and products
    # Let's assume we have 1000 users and 500 products in our ecommerce platform.
    num_users = 1000
    num_products = 500

    # Step 2: Generating the Users Data
    # Each user has an ID, age, gender, and location.
    user_data = {
        "user_id": np.arange(1, num_users + 1),  # Generate user IDs from 1 to 1000
        "age": np.random.randint(
            18, 70, size=num_users
        ),  # Random ages between 18 and 70
        "gender": np.random.choice(
            ["M", "F"], size=num_users
        ),  # Randomly assign gender as Male (M) or Female (F)
        "location": np.random.choice(
            ["Urban", "Suburban", "Rural"], size=num_users
        ),  # Randomly assign location type
    }

    # Convert the user data dictionary into a pandas DataFrame
    users_df = pd.DataFrame(user_data)

    # Step 3: Generating the Products Data
    # Each product has an ID, category, price, and rating.
    product_data = {
        "product_id": np.arange(
            1, num_products + 1
        ),  # Generate product IDs from 1 to 500
        "category": np.random.choice(
            ["Electronics", "Clothing", "Home", "Books"], size=num_products
        ),  # Randomly assign product category
        "price": np.round(
            np.random.uniform(5, 500, size=num_products), 2
        ),  # Random prices between $5 and $500, rounded to 2 decimal places
        "rating": np.round(
            np.random.uniform(1, 5, size=num_products), 1
        ),  # Random ratings between 1 and 5, rounded to 1 decimal place
    }

    # Convert the product data dictionary into a pandas DataFrame
    products_df = pd.DataFrame(product_data)

    # Step 4: Generating the User-Product Interaction Data (Purchase History or Ratings)
    # We simulate how users interact with products. For example, users can rate or buy products.

    interaction_data = {
        "user_id": np.random.choice(
            users_df["user_id"], size=5000
        ),  # Randomly select users who interacted with products
        "product_id": np.random.choice(
            products_df["product_id"], size=5000
        ),  # Randomly select products that were interacted with
        "rating": np.random.randint(
            1, 6, size=5000
        ),  # Assign random ratings (1 to 5 stars) for these interactions
        "timestamp": pd.date_range(
            start="2023-01-01", periods=5000, freq="T"
        ),  # Generate random timestamps for interactions, 1 minute apart
    }

    # Convert the interaction data dictionary into a pandas DataFrame
    interactions_df = pd.DataFrame(interaction_data)

    # Let's check the first few rows of each dataset
    users_df.head(), products_df.head(), interactions_df.head()
    return users_df, products_df, interactions_df


def preprocess_data(
    users_df: pd.DataFrame, products_df: pd.DataFrame, interactions_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Step 1: Handle missing values
    # Checking for missing values in all datasets
    print("Missing values in users data:\n", users_df.isnull().sum())
    print("Missing values in products data:\n", products_df.isnull().sum())
    print("Missing values in interactions data:\n", interactions_df.isnull().sum())

    # Step 2: Encoding categorical variables
    label_encoder = LabelEncoder()

    # Encode the gender column in users data (M -> 0, F -> 1)
    users_df["gender_encoded"] = label_encoder.fit_transform(users_df["gender"])

    # Encode the location column in users data
    users_df["location_encoded"] = label_encoder.fit_transform(users_df["location"])

    # Encode the category column in products data
    products_df["category_encoded"] = label_encoder.fit_transform(
        products_df["category"]
    )

    return users_df, products_df, interactions_df
