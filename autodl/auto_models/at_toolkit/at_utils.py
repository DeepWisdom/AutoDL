import os
from autodl.utils.log_utils import error

from ..at_toolkit.at_cons import SPEECH_TR34_PT_MODEL_PATH, SPEECH_TR34_PT_MODEL_DIR


def autodl_image_install_download():

    pass


def autodl_video_install_download():

    pass


def autodl_speech_install_download():

    os.system("apt install wget")

    if not os.path.isfile(SPEECH_TR34_PT_MODEL_PATH):
        print("Error: {} not file".format(SPEECH_TR34_PT_MODEL_PATH))

    os.system('pip install kapre==0.1.4 -i https://pypi.tuna.tsinghua.edu.cn/simple')


def autodl_nlp_install_download():

    os.system("pip install jieba_fast -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install jieba -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install pathos -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install bpemb -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("pip install keras-radam -i https://pypi.tuna.tsinghua.edu.cn/simple")
    os.system("apt-get install wget")


def autodl_tabular_install_download():

    pass


def autodl_install_download(domain):
    if domain == "image":
        autodl_image_install_download()
    elif domain == "video":
        autodl_video_install_download()
    elif domain == "speech":
        autodl_speech_install_download()
    elif domain == "nlp":
        autodl_nlp_install_download()
    elif domain == "tabular":
        autodl_tabular_install_download()
    else:
        error("Error: domain is {}, can not install_download".format(domain))
