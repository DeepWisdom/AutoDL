# coding:utf-8
import numpy as np
from ...at_toolkit.interface.adl_metadata import AdlSpeechDMetadata as ASMD

# Config
G_VAL_CL_NUM = 3
G_VAL_T_MAX_MUM = 300
G_VAL_T_MIN_NUM = 0


def listofdict_topn_sorter(raw_listofdict, attr_key, reverse=True):
    sorted_res = sorted(enumerate(raw_listofdict), key=lambda x: x[1][attr_key], reverse=reverse)
    sorted_ids = [x[0] for x in sorted_res]
    sorted_attrs = [x[1][attr_key] for x in sorted_res]
    return sorted_ids, sorted_attrs

class EvalPredSpace:
    def __init__(self):
        self.eval_pred_pool = list()
        self.g_sort_idx_pool = {"val_nauc": None, "t_acc": None, "t_loss": None}

    def put_epoch_eval_preds(self, epoch_item):
        self.eval_pred_pool.append(epoch_item)
        self.g_sort_val_nauc_idxs, self.g_sort_val_naucs = listofdict_topn_sorter(
            raw_listofdict=self.eval_pred_pool, attr_key="val_nauc"
        )

        self.g_sort_train_loss_idxs, self.g_sort_train_losss = listofdict_topn_sorter(
            raw_listofdict=self.eval_pred_pool, attr_key="t_loss", reverse=False
        )
        self.g_sort_train_acc_idxs, self.g_sort_train_accs = listofdict_topn_sorter(
            raw_listofdict=self.eval_pred_pool, attr_key="t_acc"
        )

        self.g_sort_idx_pool["val_nauc"] = self.g_sort_val_nauc_idxs
        self.g_sort_idx_pool["t_loss"] = self.g_sort_train_loss_idxs
        self.g_sort_idx_pool["t_acc"] = self.g_sort_train_acc_idxs


class EnsembleLearner:
    COMM_KEY_LIST = ["t_loss", "t_acc", "val_nauc", "val_auc", "val_acc", "val_loss"]

    def __init__(self, d_metadata: ASMD):
        self.d_metadata = d_metadata
        self.eval_pred_space = EvalPredSpace()
        self.commitee_id_pool = list()
        self.has_split_val = False

    def add_eval_pred_item(self, eval_pred_item: dict):
        self.eval_pred_space.put_epoch_eval_preds(eval_pred_item)

    def gen_committee(self, voting_conditions: dict):

        self.commitee_id_pool = list()
        for k, v in voting_conditions.items():
            assert k in self.COMM_KEY_LIST
            condition_comit_ids = self.eval_pred_space.g_sort_idx_pool.get(k)[:v]
            condition_comit_values =  [self.eval_pred_space.eval_pred_pool[i].get(k) for i in condition_comit_ids]
            if k == "t_acc" and self.eval_pred_space.g_sort_train_accs[:10].count(1) == 10:
                pass
            else:
                self.commitee_id_pool.extend(condition_comit_ids)

    def softvoting_ensemble_preds(self, voting_conditions: dict):
        self.gen_committee(voting_conditions)

        commitee_pool_preds = [
            self.eval_pred_space.eval_pred_pool[epkey].get("pred_probas") for epkey in self.commitee_id_pool
        ]
        return np.mean(commitee_pool_preds, axis=0)

    def predict_g_valid_num(self):
        def_ratio = 0.1
        g_valid_num = self.d_metadata.class_num * G_VAL_CL_NUM
        g_valid_num = max(g_valid_num, int(self.d_metadata.train_num * def_ratio))
        g_valid_num = min(g_valid_num, G_VAL_T_MAX_MUM)
        g_valid_num = max(g_valid_num, G_VAL_T_MIN_NUM)
        return g_valid_num

    def predict_if_split_val(self, token_size):
        if token_size >= self.d_metadata.train_num * 0.3:
            self.has_split_val = True
            return True
        else:
            return False


    def get_loss_godown_rate(self, g_loss_list, tail_window_size):
        tail_loss_list = g_loss_list[-tail_window_size:]
        loss_num = len(tail_loss_list)
        loss_godown_count_list = list()
        for i in range(1, loss_num):
            if tail_loss_list[i] - tail_loss_list[i - 1] < 0:
                loss_godown_count_list.append(1)
            else:
                loss_godown_count_list.append(-1)

        loss_godown_count_num = loss_godown_count_list.count(1)
        loss_godown_count_rate = round(loss_godown_count_num / (loss_num - 1), 4)

        return loss_godown_count_rate


def main():
    ensemble_learner = EnsembleLearner()


if __name__ == "__main__":
    main()
