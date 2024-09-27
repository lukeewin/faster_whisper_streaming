import collections
import faster_whisper
import numpy
import opencc
import pyaudio
import torch.cuda
import wave


# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def init_model(model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    faster_whisper_model = faster_whisper.WhisperModel(model_size_or_path=model_path, device=device, local_files_only=True)
    return faster_whisper_model


def get_levels(data, long_term_noise_level, current_noise_level):
    pegel = numpy.abs(numpy.frombuffer(data, dtype=numpy.int16)).mean()
    long_term_noise_level = long_term_noise_level * 0.995 + pegel * (1.0 - 0.995)
    current_noise_level = current_noise_level * 0.920 + pegel * (1.0 - 0.920)
    return pegel, long_term_noise_level, current_noise_level


def process_audio(model):
    while True:
        audio = pyaudio.PyAudio()
        py_stream = audio.open(rate=16000, format=pyaudio.paInt16, channels=1, input=True, frames_per_buffer=512)
        audio_buffer = collections.deque(maxlen=int((16000 // 512) * 0.5))
        frames, long_term_noise_level, current_noise_level, voice_activity_detected = [], 0.0, 0.0, False

        print("\n\nStart speaking. ", end="", flush=True)
        while True:
            data = py_stream.read(512)
            pegel, long_term_noise_level, current_noise_level = get_levels(data, long_term_noise_level, current_noise_level)
            audio_buffer.append(data)

            if voice_activity_detected:
                frames.append(data)
                if current_noise_level < ambient_noise_level + 100:
                    break  # voice actitivy ends

            if not voice_activity_detected and current_noise_level > long_term_noise_level + 300:
                voice_activity_detected = True
                print("I'm all ears.\n")
                ambient_noise_level = long_term_noise_level
                frames.extend(list(audio_buffer))

        py_stream.stop_stream(), py_stream.close(), audio.terminate()

        # Transcribe recording using whisper
        with wave.open("voice_record.wav", 'wb') as wf:
            wf.setparams((1, audio.get_sample_size(pyaudio.paInt16), 16000, 0, 'NONE', 'NONE'))
            wf.writeframes(b''.join(frames))
        segments, info = model.transcribe("voice_record.wav", without_timestamps=True)
        user_text = " ".join(seg.text for seg in segments)
        if info.language == 'zh':
            user_text = t2s.convert(user_text)
        print(f'>>>{user_text}\n<<< ', end="", flush=True)


if __name__ == '__main__':
    t2s = opencc.OpenCC('t2s.json')
    model_path = r"D:\Works\Python\Faster_Whisper\model\small"
    model = init_model(model_path)
    process_audio(model)
