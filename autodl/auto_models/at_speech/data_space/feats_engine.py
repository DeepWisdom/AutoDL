import os
from functools import partial
from itertools import repeat
import json
import multiprocessing
from multiprocessing.pool import ThreadPool, Pool
import keras
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler
from kapre.time_frequency import Melspectrogram, Spectrogram

from ...at_toolkit.interface.adl_feats_maker import AbsFeatsMaker


NCPU = multiprocessing.cpu_count()
SAMPLING_RATE = 16000
MAX_AUDIO_DURATION = 5

AUDIO_SAMPLE_RATE = 16000
KAPRE_FMAKER_WARMUP = True


def wav_to_mag_old(wav, params, win_length=400, hop_length=160, n_fft=512):
    mode = params["mode"]
    wav = extend_wav(wav, params["train_wav_len"], params["test_wav_len"], mode=mode)
    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(np.asfortranarray(linear_spect))
    mag_T = mag.T
    if mode == "test":
        mag_T = load_data(mag_T, params["train_spec_len"], params["test_spec_len"], mode)
    return mag_T


def make_kapre_mag_maker(n_fft=1024, hop_length=128, audio_data_len=80000):
    stft_model = keras.models.Sequential()
    stft_model.add(Spectrogram(n_dft=n_fft, n_hop=hop_length, input_shape=(1, audio_data_len),
                               power_spectrogram=2.0, return_decibel_spectrogram=False,
                               trainable_kernel=False, name='stft'))
    return stft_model


def wav_to_mag(wav, params, win_length=400, hop_length=160, n_fft=512):
    mode = params["mode"]
    wav = extend_wav(wav, params["train_wav_len"], params["test_wav_len"], mode=mode)
    wav2feat_mode = 1

    if wav2feat_mode == 0:
        pass
    elif wav2feat_mode == 1:
        linear_sft = librosa.stft(np.asfortranarray(wav), n_fft=n_fft, win_length=win_length, hop_length=hop_length)  # linear spectrogram
        mag_T = np.abs(linear_sft)
        pass

    elif wav2feat_mode == 2:
        linear_sft = librosa.stft(np.asfortranarray(wav), n_fft=n_fft, win_length=win_length,
                                  hop_length=hop_length)  # linear spectrogram
        mag_T = linear_sft

    if mode == "test":
        mag_T = load_data(mag_T, params["train_spec_len"], params["test_spec_len"], mode)
    return mag_T


def get_fixed_array(X_list, len_sample=5, sr=SAMPLING_RATE):
    for i in range(len(X_list)):
        if len(X_list[i]) < len_sample * sr:
            n_repeat = np.ceil(sr * len_sample / X_list[i].shape[0]).astype(np.int32)
            X_list[i] = np.tile(X_list[i], n_repeat)

        X_list[i] = X_list[i][: len_sample * sr]

    X = np.asarray(X_list)
    X = np.stack(X)
    X = X[:, :, np.newaxis]
    X = X.transpose(0, 2, 1)
    return X

def mel_feats_transform(x_mel):
    x_feas = []
    for i in range(len(x_mel)):
        mel = np.mean(x_mel[i], axis=0).reshape(-1)
        mel_std = np.std(x_mel[i], axis=0).reshape(-1)
        fea_item = np.concatenate([mel, mel_std], axis=-1)
        x_feas.append(fea_item)

    x_feas = np.asarray(x_feas)
    scaler = StandardScaler()
    X = scaler.fit_transform(x_feas[:, :])
    return X

def extract_parallel(data, extract):
    data_with_index = list(zip(data, range(len(data))))
    results_with_index = list(pool.map(extract, data_with_index))
    results_with_index.sort(key=lambda x: x[1])
    results = []
    for res, idx in results_with_index:
        results.append(res)

    return np.asarray(results)

def extract_for_one_sample(tuple, extract, use_power_db=False, **kwargs):
    data, idx = tuple
    r = extract(data, **kwargs)
    if use_power_db:
        r = librosa.power_to_db(r)

    r = r.transpose()
    return r, idx

def extend_wav(wav, train_wav_len=40000, test_wav_len=40000, mode="train"):
    if mode == "train":
        div, mod = divmod(train_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        if np.random.random() < 0.3:
            extended_wav = extended_wav[::-1]
        return extended_wav
    else:
        div, mod = divmod(test_wav_len, wav.shape[0])
        extended_wav = np.concatenate([wav] * div + [wav[:mod]])
        return extended_wav

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(
        np.asfortranarray(wav), n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    return linear.T


def load_data(mag, train_spec_len=250, test_spec_len=250, mode="train"):
    freq, time = mag.shape
    if mode == "train":
        if time - train_spec_len > 0:
            randtime = np.random.randint(0, time - train_spec_len)
            spec_mag = mag[:, randtime : randtime + train_spec_len]
        else:
            spec_mag = mag[:, :train_spec_len]
    else:
        spec_mag = mag[:, :test_spec_len]
    mu = np.mean(spec_mag, 0, keepdims=True)
    std = np.std(spec_mag, 0, keepdims=True)
    return (spec_mag - mu) / (std + 1e-5)


class KapreMelSpectroGramFeatsMaker(AbsFeatsMaker):
    SAMPLING_RATE = 16000
    N_MELS = 30
    HOP_LENGTH = int(SAMPLING_RATE * 0.04)
    N_FFT = 1024
    FMIN = 20
    FMAX = SAMPLING_RATE // 2

    CROP_SEC = 5

    def __init__(self, feat_name, feat_tool="Kapre"):
        super().__init__(feat_tool, feat_name)
        self.kapre_melspectrogram_extractor = None
        self.kape_params = {
            "SAMPLING_RATE": self.SAMPLING_RATE,
            "N_MELS": self.N_MELS,
            "HOP_LENGTH": int(self.SAMPLING_RATE * 0.04),
            "N_FFT": self.N_FFT,
            "FMIN": self.FMIN,
            "FMAX": self.SAMPLING_RATE // 2,
            "CROP_SEC": self.CROP_SEC,
        }
        self.init_kapre_melspectrogram_extractor()

    def make_melspectrogram_extractor(self, input_shape, sr=SAMPLING_RATE):
        model = keras.models.Sequential()
        model.add(
            Melspectrogram(
                fmax=self.kape_params.get("FMAX"),
                fmin=self.kape_params.get("FMIN"),
                n_dft=self.kape_params.get("N_FFT"),
                n_hop=self.kape_params.get("HOP_LENGTH"),
                n_mels=self.kape_params.get("N_MELS"),
                name="melgram",
                image_data_format="channels_last",
                input_shape=input_shape,
                return_decibel_melgram=True,
                power_melgram=2.0,
                sr=sr,
                trainable_kernel=False,
            )
        )
        return model

    def init_kapre_melspectrogram_extractor(self):
        self.kapre_melspectrogram_extractor = self.make_melspectrogram_extractor(
            (1, self.kape_params.get("CROP_SEC") * self.kape_params.get("SAMPLING_RATE"))
        )
        if KAPRE_FMAKER_WARMUP:
            warmup_size = 10
            warmup_x = [
                np.array([np.random.uniform() for i in range(48000)], dtype=np.float32) for j in range(warmup_size)
            ]
            warmup_x_mel = self.make_features(warmup_x, feats_maker_params={"len_sample": 5, "sr": 16000})

    def make_features(self, raw_data, feats_maker_params: dict):
        raw_data = [sample[0 : MAX_AUDIO_DURATION * AUDIO_SAMPLE_RATE] for sample in raw_data]

        X = get_fixed_array(raw_data, len_sample=feats_maker_params.get("len_sample"), sr=feats_maker_params.get("sr"))
        X = self.kapre_melspectrogram_extractor.predict(X)

        X = np.squeeze(X)
        X = X.transpose(0, 2, 1)

        X = mel_feats_transform(X)
        return X


class LibrosaMelSpectroGramFeatsMaker(AbsFeatsMaker):
    FFT_DURATION = 0.1
    HOP_DURATION = 0.04

    def __init__(self, feat_name, feat_tool="Librosa"):
        super().__init__(feat_tool, feat_name)

    def extract_melspectrogram_parallel(
        self, data, sr=16000, n_fft=None, hop_length=None, n_mels=30, use_power_db=False
    ):
        if n_fft is None:
            n_fft = int(sr * self.FFT_DURATION)
        if hop_length is None:
            hop_length = int(sr * self.HOP_DURATION)
        extract = partial(
            extract_for_one_sample,
            extract=librosa.feature.melspectrogram,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            use_power_db=use_power_db,
        )
        results = extract_parallel(data, extract)

        return results

    def make_features(self, raw_data, feats_maker_params: dict):
        x_mel = self.extract_melspectrogram_parallel(raw_data, n_mels=30, use_power_db=True)
        # tranform melspectrogram features.
        x_mel_transformed = mel_feats_transform(x_mel)
        return x_mel_transformed


pool = Pool(os.cpu_count())


class LbrsTr34FeatsMaker(AbsFeatsMaker):
    def __init__(self, feat_tool, feat_name):
        super().__init__(feat_tool, feat_name)
        self.feat_des = "for_TR34"

    def pre_trans_wav_update(self, wav_list, params):
        if len(wav_list) == 0:
            return []
        elif len(wav_list) > NCPU * 2:
            mag_arr = pool.starmap(wav_to_mag, zip(wav_list, repeat(params)))
            return mag_arr
        else:
            mag_arr = [wav_to_mag(wav, params) for wav in wav_list]
            return mag_arr

    def make_features(self, raw_data, feats_maker_params: dict):
        tr34_data_features = self.pre_trans_wav_update(raw_data, feats_maker_params)
        return tr34_data_features
