from surprise import accuracy


def eval_model(predictions):
    mae = accuracy.mae(predictions)
    rmse = accuracy.rmse(predictions)

    performance_report = {"RMSE": rmse, "MAE": mae}

    print("Model Performance Report:")
    for metric, score in performance_report.items():
        print(f"{metric}: {score:.4f}")
