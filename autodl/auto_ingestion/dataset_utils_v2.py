from autodl.auto_ingestion import data_io


def get_dataset_basename(dataset_dir: str):
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    basename = datanames[0]
    #### INVENTORY DATA (and sort dataset names alphabetically)
    datanames = data_io.inventory_data(dataset_dir)
    #### Delete zip files and metadata file
    datanames = [x for x in datanames if x.endswith('.data')]

    basename = datanames[0]

    return basename
