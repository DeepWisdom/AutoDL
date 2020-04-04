import os

ADL_ROOT_PATH = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
print(ADL_ROOT_PATH)

TR34_FN = "thin_resnet34.h5"

SPEECH_TR34_PT_MODEL_DIR = os.path.join(ADL_ROOT_PATH, "at_speech", "pretrained_models")
SPEECH_TR34_PT_MODEL_PATH = os.path.join(SPEECH_TR34_PT_MODEL_DIR, TR34_FN)

