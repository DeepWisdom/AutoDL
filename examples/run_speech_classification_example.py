import os

from autodl import Model, AutoDLDataset
from autodl.auto_ingestion import dataset_utils_v2
from autodl.auto_models.at_speech.model import Model as SpeechModel
from autodl.auto_ingestion.pure_model_run import run_single_model

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def do_speech_classification_demo():
    remaining_time_budget = 1200
    max_epoch = 100

    # Speech formmated datasets.
    dataset_dir = "ADL_sample_data/data01_formmated"
    basename = dataset_utils_v2.get_dataset_basename(dataset_dir)
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    model = SpeechModel(D_train.get_metadata())  # The metadata of D_train and D_test only differ in sample_count

    run_single_model(model, dataset_dir, basename, remaining_time_budget, max_epoch)


def main():
    do_speech_classification_demo()


if __name__ == '__main__':
    main()
