import tensorflow as tf
import argparse
import numpy as np
import numpy.typing as npt
import sounddevice as sd
from time import time
import psutil
import redis
import time
import uuid
import argparse


SAMPLING_RATE = 16000
CHANNELS = 1
RESOLUTION = "int16"
BLOCKSIZE = int(0.5 * SAMPLING_RATE)
audio_buffer = np.zeros(shape=(SAMPLING_RATE, CHANNELS))

def monitor(prev_state, state):
    global prev_time
    
    delta = np.inf
    if prev_state==1:
        delta = time.time() - prev_time
        
    if state==1 and delta >= 1:
        timestamp = time.time()
        prev_time = timestamp
        timestamp_ms = int(timestamp * 1000)
        battery_level = psutil.sensors_battery().percent
        power_plugged = int(psutil.sensors_battery().power_plugged)

        redis_client.ts().add(f'{mac_address}:battery', timestamp_ms, battery_level) # Add the value in the timeseries every second

        redis_client.ts().add(f'{mac_address}:power', timestamp_ms, power_plugged) # Add the value in the timeseries every second
        



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

class MFCC():
    def __init__(
        self, 
        sampling_rate: int,
        frame_length_in_s: float,
        frame_step_in_s: float,
        num_mel_bins: int,
        lower_frequency: int,
        upper_frequency: int,
        num_coefficients: int
    ):
        self.log_mel_spectrogram_processor = MelSpectrogram(sampling_rate,frame_length_in_s, frame_step_in_s, num_mel_bins,lower_frequency,upper_frequency)
        self.num_coefficients = num_coefficients

    def get_mfccs(self, audio):
        log_mel_spectrogram = self.log_mel_spectrogram_processor.get_mel_spec(audio)
        mfcc = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        return mfcc[:self.num_coefficients]

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
    global is_audio_buffer_silent, audio_buffer, state

    # Put the last 0.5s at the beginning of the buffer and the latest at the end.
    audio_buffer = np.roll(audio_buffer, -BLOCKSIZE)
    audio_buffer[BLOCKSIZE:, :] = indata

    is_audio_buffer_silent = voice_activity_detector.is_silence(audio_buffer)

    if not is_audio_buffer_silent:
        # audio not silent, so we store it.
        audio = preprocess_audio(audio_buffer)
        new_state = classification(interpreter=interpreter, input_details=input_details, output_details=output_details, current_state=state, processor=mfcc_processor, audio=audio)
        monitor(state, new_state)
        
        if(new_state == 1):
            print('...Monitoring')
        else:
            print('...Not Monitoring')
    


PREPROCESSING_ARGS_MFCC = {
    'sampling_rate': 16000,
    'frame_length_in_s': 0.032,
    'frame_step_in_s': 0.016,
    'num_mel_bins': 64,
    'lower_frequency': 0,
    'upper_frequency': 8000,
    'num_coefficients': 64
}

mfcc_processor = MFCC(**PREPROCESSING_ARGS_MFCC)
    
interpreter = tf.lite.Interpreter(model_path='tflite_models/model12.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def classification(interpreter, input_details, output_details, current_state, processor, audio):
    
    mfccs = processor.get_mfccs(audio)
    mfccs = tf.expand_dims(mfccs, 0)
    mfccs = tf.expand_dims(mfccs, -1)
    interpreter.set_tensor(input_details[0]['index'], mfccs)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    prob_yes, prob_no = output[0][0], output[0][1]
    
    if prob_yes > 0.99: # Start monitoring
        return 1
    if prob_no > 0.99: # Stop monitoring
        return 0
    
    print('NOT A COMMAND')
    return current_state # Remain in current state
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--host", type = str)
    parser.add_argument("--port", type=str)
    parser.add_argument("--user", type=str)
    parser.add_argument("--password", type=str)
    args = parser.parse_args()


    REDIS_HOST = args.host
    REDIS_PORT = args.port
    REDIS_USERNAME = args.user
    REDIS_PASSWORD = args.password


    redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, username=REDIS_USERNAME, password=REDIS_PASSWORD)
    is_connected = redis_client.ping()
    print('Redis Connected:', is_connected)

    mac_address = hex(uuid.getnode())
    
    try:
        redis_client.ts().create(f'{mac_address}:battery')
    except redis.ResponseError:
        pass

    try:
        redis_client.ts().create(f'{mac_address}:power')
    except redis.ResponseError:
        pass

        
    state = 0
    prev_time = time.time()
    
    is_audio_buffer_silent = True
    # Instance of VAD with the parameters found in exercise 1.1
    voice_activity_detector = VAD(
        sampling_rate=16000,
        frame_length_in_s=0.032,
        num_mel_bins=80,
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
                break
    
        

