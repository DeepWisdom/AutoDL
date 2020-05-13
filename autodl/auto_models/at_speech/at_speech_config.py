from ..at_speech.at_speech_cons import CLS_LR_LIBLINEAER, CLS_LR_SAG, CLS_TR34



# for data01
MODEL_SELECT_DEF = {
    0: CLS_LR_LIBLINEAER,
    1: CLS_LR_LIBLINEAER,
    2: CLS_TR34,
}


TR34_TRAINPIP_WARMUP = 2
IF_VAL_ON = False


TFDS2NP_TAKESIZE_RATION_LIST = [0.1, 0.2, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

IF_TRAIN_BREAK_CONDITION = False


class Tr34SamplerHpParams:
    SAMPL_PA_F_PERC_NUM = 10
    SAMPL_PA_F_MAX_NUM = 200
    SAMPL_PA_F_MIN_NUM = 200


class ThinRes34Config(object):
    ENABLE_CB_LRS = True
    ENABLE_CB_ES = True

    Epoch = 1
    VERBOSE = 2
    MAX_SEQ_NUM_CUT = 140000

    MAX_ROUND_NUM = 3000
    MAX_LEFT_TIME_BUD = 10
    TR34_INIT_WD = 1e-3
    INIT_BRZ_L_NUM = 124
    INIT_BRZ_L_NUM_WILD = 100
    PRED_SIZE = 8
    CLASS_NUM_THS = 37

    ENABLE_PRE_ENSE = True
    ENS_TOP_VLOSS_NUM = 8
    ENS_TOP_VACC_NUM = 0

    MAX_BATCHSIZE = 32
    FIRST_ROUND_EPOCH = 8
    LEFT_ROUND_EPOCH = 1


    TR34_INIT_LR = 0.00175
    STEP_DE_LR = 0.002
    MAX_LR = 1e-3 * 1.5
    MIN_LR = 1e-4 * 5

    FULL_VAL_R_START = 2500
    G_VAL_CL_NUM = 3
    G_VAL_T_MAX_MUM = 0
    G_VAL_T_MIN_NUM = 0

    HIS_METRIC_SHOW_NUM = 10

    FE_RS_SPEC_LEN_CONFIG = {
        1: (125, 125, 0.002),
        5: (250, 250, 0.003),
        20: (1500, 1500, 0.004),
        50: (2250, 2250, 0.004),
    }

    FE_RS_SPEC_LEN_CONFIG_AGGR = {
        1: (150, 150, 0.002),
        10: (500, 500, 0.003),
        21: (1500, 1500, 0.004),
        50: (2250, 2250, 0.004),
    }

    FE_RS_SPEC_LEN_CONFIG_MILD = {
        1: (350, 350, 0.002),
        10: (500, 500, 0.004),
        20: (1000, 1000, 0.002),
        50: (1500, 1500, 0.004),
    }
