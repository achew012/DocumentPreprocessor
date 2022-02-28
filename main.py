from clearml import PipelineController

PIPELINE_PROJECT_NAME = "TextPreprocessing"
PIPELINE_NAME = "raw-denoiser-auditor-datasets"

ETL_TASK_PROJECT_NAME = "datasets/c4"
DENOISER_TASK_PROJECT_NAME = "DocumentProcessing"
DATASET_AUDIT_PROJECT_NAME = "DataAudit"


pipe = PipelineController(
    project=PIPELINE_PROJECT_NAME,
    name=PIPELINE_NAME,
    version="0.1",
    add_pipeline_tags=True,
)
pipe.set_default_execution_queue("compute")  # set to queue with GPU

# pipe.add_step(
#     name="stage_data",
#     base_task_project="datasets/c4",
#     base_task_name="dataset_store_c4_dataset",
# )

# pipe.add_step(
#     name="dataset_etl",
#     base_task_project=ETL_TASK_PROJECT_NAME,
#     base_task_name="dataset_load_csv",
# )

pipe.add_step(
    name="denoiser",
    base_task_project=DENOISER_TASK_PROJECT_NAME,
    base_task_name="LED-Denoiser-train",
    parameter_override={
        # "General/clearml_dataset_project_name": ${dataset_etl.artifacts.data.url}
        "General/clearml_dataset_project_name": ETL_TASK_PROJECT_NAME,
        "General/clearml_dataset_name": "dataset_load_csv",
    },
)

pipe.add_step(
    name="dataset_audit",
    base_task_project=DATASET_AUDIT_PROJECT_NAME,
    base_task_name="dataset_audit",
    parameter_override={
        "General/source_path": ${denoiser.artifacts.data.url},
    },
)

# Starting the pipeline (in the background)
pipe.start()

print("done")
