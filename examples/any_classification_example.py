import os

from autodl import Model, AutoDLDataset
from autodl.AutoDL_ingestion_program import dataset_utils_v2
from autodl.AutoDL_scoring_program.score import autodl_auc, accuracy, get_solution

os.environ["CUDA_VISIBLE_DEVICES"]='0'

def do_any_classification_demo():
    remaining_time_budget = 1200
    max_loop = 100

    # Text
    dataset_dir = "ADL_sample_data/imdb_data"

    basename = dataset_utils_v2.get_dataset_basename(dataset_dir)
    D_train = AutoDLDataset(os.path.join(dataset_dir, basename, "train"))
    D_test = AutoDLDataset(os.path.join(dataset_dir, basename, "test"))
    solution = get_solution(solution_dir=dataset_dir)

    M = Model(D_train.get_metadata())  # The metadata of D_train and D_test only differ in sample_count

    for i in range(max_loop):
        M.train(D_train.get_dataset(),
                remaining_time_budget=remaining_time_budget)

        Y_pred = M.test(D_test.get_dataset(),
                        remaining_time_budget=remaining_time_budget)

        # Evaluation.
        nauc_score = autodl_auc(solution=solution, prediction=Y_pred)
        acc_score = accuracy(solution=solution, prediction=Y_pred)

        print(Y_pred)
        print("Loop={}, evaluation: nauc_score={}, acc_score={}".format(i, nauc_score, acc_score))


def main():
    do_any_classification_demo()


if __name__ == '__main__':
    main()
