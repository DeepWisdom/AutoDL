from autodl.convertor import autonlp_2_autodl_format, autospeech_2_autodl_format


def convertor_nlp_demo():
    raw_autonlp_datadir = "~/AutoNLP/AutoDL_sample_data/O1"
    autonlp_2_autodl_format(input_dir=raw_autonlp_datadir)

def convertor_speech_demo():
    raw_autospeech_datadir = "~/AutoSpeech/AutoDL_sample_data/data01"
    autospeech_2_autodl_format(input_dir=raw_autospeech_datadir)

def main():
    convertor_nlp_demo()
    convertor_speech_demo()

if __name__ == '__main__':
    main()