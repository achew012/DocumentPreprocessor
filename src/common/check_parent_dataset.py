from clearml import Dataset


def create_dataset(dataset_project, dataset_name):
    parent_dataset = _get_last_child_dataset(dataset_project, dataset_name)
    if parent_dataset:
        print("create child")
        if not parent_dataset.is_final():
            parent_dataset.finalize()
        child_dataset = Dataset.create(
            dataset_name, dataset_project, parent_datasets=[parent_dataset]
        )
        return child_dataset
    else:
        print("create parent")
        dataset = Dataset.create(dataset_name, dataset_project)
        return dataset


def _get_last_child_dataset(dataset_project, dataset_name):
    datasets_dict = Dataset.list_datasets(
        dataset_project=dataset_project, partial_name=dataset_name, only_completed=False
    )
    if datasets_dict:
        datasets_dict_latest = datasets_dict[-1]
        return Dataset.get(dataset_id=datasets_dict_latest["id"])
