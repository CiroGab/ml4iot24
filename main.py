## EX 1.2 HW 1

import tensorflow as tf
import tensorflow_io as tfio


class AudioReader():
    def __init__(self, resolution, sampling_rate):
        self.resolution = resolution
        self.sampling_rate = sampling_rate

    def get_audio(self, filename):
        audio_io_tensor = tfio.audio.AudioIOTensor(filename, self.resolution)        

        audio_tensor = audio_io_tensor.to_tensor()
        audio_tensor = tf.squeeze(audio_tensor)

        audio_float32 = tf.cast(audio_tensor, tf.float32)
        audio_normalized = audio_float32 / self.resolution.max

        zero_padding = tf.zeros(self.sampling_rate - tf.shape(audio_normalized), dtype=tf.float32)
        audio_padded = tf.concat([audio_normalized, zero_padding], axis=0)

        return audio_padded


class Spectrogram():
    def __init__(self, sampling_rate, frame_length_in_s, frame_step_in_s):
        self.frame_length = int(frame_length_in_s * sampling_rate)
        self.frame_step = int(frame_step_in_s * sampling_rate)

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(
            audio, 
            frame_length=self.frame_length,
            frame_step=self.frame_step,
            fft_length=self.frame_length
        )
        spectrogram = tf.abs(stft)

        return spectrogram



class MelSpectrogram():
    def __init__(
        self, 
        sampling_rate,
        frame_length_in_s,
        frame_step_in_s,
        num_mel_bins,
        lower_frequency,
        upper_frequency
    ):
        self.spectrogram_processor = Spectrogram(sampling_rate, frame_length_in_s, frame_step_in_s)
        num_spectrogram_bins = self.spectrogram_processor.frame_length // 2 + 1

        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=num_mel_bins,
            num_spectrogram_bins=num_spectrogram_bins,
            sample_rate=sampling_rate,
            lower_edge_hertz=lower_frequency,
            upper_edge_hertz=upper_frequency
        )

    def get_mel_spec(self, audio):
        spectrogram = self.spectrogram_processor.get_spectrogram(audio)
        mel_spectrogram = tf.matmul(spectrogram, self.linear_to_mel_weight_matrix)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)

        return log_mel_spectrogram

 
