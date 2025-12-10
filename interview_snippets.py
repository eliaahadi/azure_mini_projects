# 1.1. Interview-style prompt

# “Given a CSV file in blob storage, show how you would:
# 	1.	Load it,
# 	2.	Train a simple binary classifier, and
# 	3.	Register the model with a new version in a model registry.”

'''
pseudo code:

functon train_register_model(storage_path, registry_path):

# 1. Load data
data_csv = download_from_storage(storage_path)
df = read_csv(data_csv)
X, y = split_features_target(df, target_col = 'target')

# 2. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 3. Build pipeline
pipeline = Pipeline(
    scaler = Standardscaler()
    model = Logisticsregression()
)

pipeline.fit(X_train, y_train)

# 4. Evaluate
y_pred, y_probab = pipeine.predict(X_test), pipeline.predict_probab(X_test)
metrics = compute_metrics(y_test, y_pred, y_probab)

# 5. Save artifacts
save_to_disk(pipeline, joblib)
save_to_disk(metrics, json)

# 6. Register model
version_dir
copy
write_metadata

return metrics, version_dir

'''


'''
Snippet 2 – Batch scoring a file with latest model

2.1. Interview-style prompt

“Show how you would implement a batch scoring script that:
	•	Loads the latest registered model,
	•	Reads an input CSV,
	•	Writes predictions and scores to an output CSV.”


'''


'''Snippet 3 – Simple FastAPI predict endpoint

3.1. Interview-style prompt

“Implement an HTTP endpoint for real-time scoring:
	•	It accepts a list of feature dictionaries in JSON,
	•	Uses a loaded model,
	•	Returns predictions and probabilities for each record.”
'''



'''
Snippet 4 – Simple “orchestrator” for daily batch (ADF-like)

4.1. Interview-style prompt

“Sketch how you’d orchestrate a nightly batch scoring job:
	•	Given a date,
	•	Read the correct input file,
	•	Call a batch scoring function,
	•	Log the run result.”
'''