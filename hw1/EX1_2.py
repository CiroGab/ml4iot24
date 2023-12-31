import tensorflow as tf
import argparse
import numpy as np
import numpy.typing as npt
import os
import sounddevice as sd
from time import time
from scipy.io.wavfile import write

SAMPLING_RATE = 16000
CHANNELS = 1
RESOLUTION = "int16"
BLOCKSIZE = int(0.5 * SAMPLING_RATE)
audio_buffer = np.zeros(shape=(SAMPLING_RATE, CHANNELS))


def preprocess_audio(indata: npt.ArrayLike) -> tf.Tensor:
    """Return the preprocessed audio."""
    tf_indata = tf.convert_to_tensor(indata, dtype=tf.float32)
    audio_tensor = tf.squeeze(tf_indata)
    audio_normalized = audio_tensor / tf.int16.max

    return audio_normalized


class Spectrogram:
    def __init__(
        self,
        sampling_rate: int,
        frame_length_in_s: float,
        frame_step_in_s: float,
    ):
        """Initialize frame lenght and frame step."""
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio: tf.Tensor) -> tf.Tensor:
        """Return the spectogram of a given audio."""
        stft = tf.signal.stft(
            audio,
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length,
        )
        spectrogram = tf.abs(stft)

        return spectrogram


class MelSpectrogram:
    def __init__(
        self,
        sampling_rate: int,
        frame_length_in_s: float,
        frame_step_in_s: float,
        num_mel_bins: int,
        lower_frequency: int,
        upper_frequency: int,
    ):
        """Initialize spectogram processor and the linear_to_mel_matrix."""
        self.spectrogram_processor = Spectrogram(
            sampling_rate, frame_length_in_s, frame_step_in_s
        )
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = (
            tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins=num_mel_bins,
                num_spectrogram_bins=num_spectrogram_bins,
                sample_rate=sampling_rate,
                lower_edge_hertz=lower_frequency,
                upper_edge_hertz=upper_frequency,
            )
        )

    def get_mel_spec(self, audio: tf.Tensor) -> tf.Tensor:
        """Return the mel spectogram."""
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(
            spectrogram, self.linear_to_mel_weight_matrix
        )
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.0e-6)

        return log_mel_spectrogram


class VAD:
    def __init__(
        self,
        sampling_rate: int,
        frame_length_in_s: float,
        num_mel_bins: int,
        lower_frequency: int,
        upper_frequency: int,
        dbFSthres: int,
        duration_thres: float,
    ):
        """Initialize the mel spectogram processor and the two thresholds."""
        self.frame_length_in_s = frame_length_in_s
        self.mel_spec_processor = MelSpectrogram(
            sampling_rate,
            frame_length_in_s,
            frame_length_in_s,
            num_mel_bins,
            lower_frequency,
            upper_frequency,
        )
        self.dbFSthres = dbFSthres
        self.duration_thres = duration_thres

    def is_silence(self, audio: npt.ArrayLike) -> bool:
        """Return if the audio is silent or not."""
        audio = preprocess_audio(indata=audio)
        log_mel_spec = self.mel_spec_processor.get_mel_spec(audio)
        dbFS = 20 * log_mel_spec
        energy = tf.math.reduce_mean(dbFS, axis=1)

        non_silence = energy > self.dbFSthres
        non_silence_frames = tf.math.reduce_sum(
            tf.cast(non_silence, tf.float32)
        )
        non_silence_duration = (non_silence_frames + 1) * self.frame_length_in_s

        if non_silence_duration > self.duration_thres:
            return False
        else:
            return True


def callback(indata: npt.ArrayLike, frames, callback_time, status):
    """This is called (from a separate thread) for each audio block."""
    global is_audio_buffer_silent, audio_buffer

    # Put the last 0.5s at the beginning of the buffer and the latest at the end.
    audio_buffer = np.roll(audio_buffer, -BLOCKSIZE)
    audio_buffer[BLOCKSIZE:, :] = indata

    is_audio_buffer_silent = voice_activity_detector.is_silence(audio_buffer)

    if not is_audio_buffer_silent:
        # audio not silent, so we store it.
        timestamp = time()
        write(f"{timestamp}.wav", SAMPLING_RATE, audio_buffer)
        filesize_in_bytes = os.path.getsize(f"{timestamp}.wav")
        filesize_in_kb = filesize_in_bytes / 1024
        print(f"Size: {filesize_in_kb:.2f}KB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    is_audio_buffer_silent = True
    # Instance of VAD with the parameters found in exercise 1.1
    voice_activity_detector = VAD(
        sampling_rate=16000,
        frame_length_in_s=0.032,
        num_mel_bins=12,
        lower_frequency=0,
        upper_frequency=8000,
        dbFSthres=-40,
        duration_thres=0.05,
    )

    with sd.InputStream(
        device=args.device,
        channels=CHANNELS,
        dtype=RESOLUTION,
        samplerate=SAMPLING_RATE,
        blocksize=BLOCKSIZE,
        callback=callback,
    ):
        while True:
            key = input()
            if key in ("q", "Q"):
                print("Stop recording.")
                break
