# This is conceptual / skeleton code

from azure.identity import DefaultAzureCredential
from azure.ai.ml import MLClient, command
from azure.ai.ml.entities import AmlCompute

subscription_id = "<YOUR_SUBSCRIPTION_ID>"
resource_group = "<YOUR_RG>"
workspace_name = "<YOUR_WORKSPACE>"

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id,
    resource_group,
    workspace_name,
)

# Optional: create compute cluster
compute_name = "cpu-cluster"
try:
    ml_client.compute.get(compute_name)
except Exception:
    compute = AmlCompute(
        name=compute_name,
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=2,
    )
    ml_client.compute.begin_create_or_update(compute).result()

# Define a command job that runs your train script
job = command(
    code=".",  # repo root containing day3_train_local_mlflow.py
    command="python day3_train_local_mlflow.py",
    environment="AzureML-sklearn-1.0-ubuntu20.04-py38-cpu",  # or custom env
    compute=compute_name,
    experiment_name="day3_azure_experiment",
    display_name="day3-logreg-baseline",
)

returned_job = ml_client.jobs.create_or_update(job)
print("Job submitted. Name:", returned_job.name)