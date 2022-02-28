from clearml import Task, StorageManager, Logger, Dataset

PROJECT_NAME = "incubation c4"
TASK_NAME = "dataset_store_c4_dataset"

task = Task.init(project_name=PROJECT_NAME, task_name=TASK_NAME)
task.set_base_docker(
    docker_image="nvidia/cuda:11.4.0-cudnn8-devel-ubuntu20.04",
)

args = {
    "clean_dataset_name": "c4",
    "clean_dataset_variant_name": "en",
    "clean_dataset_split": "train",
    "unclean_dataset_name": "c4",
    "unclean_dataset_variant_name": "en.noclean",
    "unclean_dataset_split": "train",
}

task.connect(args)
task.execute_remotely(queue_name='cpu-only')

print(args)

clean_dataset_name = args["clean_dataset_name"]
clean_dataset_variant_name = args["clean_dataset_variant_name"]
clean_dataset_split = args["clean_dataset_split"]

unclean_dataset_name = args["unclean_dataset_name"]
unclean_dataset_variant_name = args["unclean_dataset_variant_name"]
unclean_dataset_split = args["unclean_dataset_split"]


################## dataset_store_hf_datasets #################
from datasets import load_dataset, load_from_disk, concatenate_datasets

cleaned_dataset = load_dataset(
    path=clean_dataset_name, name=clean_dataset_variant_name, split=clean_dataset_split,data_files='https://huggingface.co/datasets/allenai/c4/blob/main/en/c4-train.00000-of-01024.json.gz'
)

uncleaned_dataset = load_dataset(
    path=unclean_dataset_name, name=unclean_dataset_variant_name, split=unclean_dataset_split,data_files='https://huggingface.co/datasets/allenai/c4/blob/main/en.noclean/c4-train.00000-of-07168.json.gz'
)

print("Number of samples in {} dataset".format(args['clean_dataset_name']), cleaned_dataset.num_rows)
print("Number of samples in {} dataset".format(args['unclean_dataset_name']), uncleaned_dataset.num_rows)


def dataset_to_shard(dataset, shard_path="/tmp/dataset", num_shards=8):
    for index in range(num_shards):
        dataset.shard(num_shards=num_shards, index=index).save_to_disk(
            "{}/shard_{}".format(shard_path, index)
        )
    return shard_path


def shard_to_dataset(shard_path="/tmp/dataset", num_shards=8):
    dataset = concatenate_datasets(
        [load_from_disk("{}/shard_{}".format(shard_path, i)) for i in range(num_shards)]
    )
    return dataset

def join(clean_set, uncleaned_set):

    print('test')

    return dataset


## shard dataset and save
cleaned_shard_path = dataset_to_shard(cleaned_dataset,shard_path="/tmp/cleaned_dataset")
uncleaned_shard_path = dataset_to_shard(uncleaned_dataset,shard_path="/tmp/uncleaned_dataset")

## load dataset from shards
clean_dataset = shard_to_dataset(cleaned_shard_path,shard_path="/tmp/cleaned_dataset")
unclean_dataset = shard_to_dataset(uncleaned_shard_path,shard_path="/tmp/uncleaned_dataset")

print(clean_dataset.features)
print(clean_dataset.unique('url'))

## save dataset shards as Dataset
# clean_dataset_name = "_".join([clean_dataset_name, clean_dataset_variant_name, clean_dataset_split])
# dataset = Dataset.create(
#     dataset_project=PROJECT_NAME,
#     dataset_name=clean_dataset_name,
#     dataset_tags=["huggingface", clean_dataset_name],
# )
# dataset.add_files(cleaned_shard_path)
# dataset.upload()
# dataset.finalize()

# unclean_dataset_name = "_".join([unclean_dataset_name, unclean_dataset_variant_name, unclean_dataset_split])
# dataset = Dataset.create(
#     dataset_project=PROJECT_NAME,
#     dataset_name=unclean_dataset_name,
#     dataset_tags=["huggingface", unclean_dataset_name],
# )
# dataset.add_files(uncleaned_shard_path)
# dataset.upload()
# dataset.finalize()

# # we are done
# print("Done")