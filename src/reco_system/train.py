import pickle

import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as surprise_train_test_split

from reco_system.dataset import get_dataset, preprocess_data


def train_model(interactions_df: pd.DataFrame) -> None:
    # Step 1: Prepare the data for Surprise
    reader = Reader(
        rating_scale=(1, 5)
    )  # The rating scale in our dataset is from 1 to 5
    data = Dataset.load_from_df(
        interactions_df[["user_id", "product_id", "rating"]], reader
    )

    # Step 2: Train-test split
    trainset, testset = surprise_train_test_split(data, test_size=0.2)

    # Step 3: Train the SVD model
    model = SVD()  # Initialize the SVD model
    model.fit(trainset)  # Train the model on the training set

    # Step 4: Test the model on the test set
    predictions = model.test(testset)

    # Step 5: Evaluate the performance using RMSE
    rmse = accuracy.rmse(predictions)
    print(f"RMSE: {rmse}")

    # Step 1: Save the trained SVD model to a file
    model_filename = "./src/model/svd_model.pkl"
    with open(model_filename, "wb") as model_file:
        pickle.dump(model, model_file)

    print(f"Model saved to {model_filename}")


if __name__ == "__main__":
    users_df, products_df, interactions_df = get_dataset()
    _, _, interactions_df = preprocess_data(users_df, products_df, interactions_df)
    train_model(interactions_df)
