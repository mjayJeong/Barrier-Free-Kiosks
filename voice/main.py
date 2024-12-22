import subprocess
import tempfile
import soundfile as sf
from faster_whisper import WhisperModel
import numpy as np
import os
from rapidfuzz import process
import time
from sklearn.metrics.pairwise import cosine_similarity
import pyaudio
import wave


words_list = ["주민등록","등본","확인","법원","교육","공공","부동산","금융","병원",
              "전체포함","전체미포함","예","아니","카드","현금","IC","모바일","발급",
              "영","하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉",
              "공","일","이","삼","사","오","육","칠","팔","구",
              "한부","두부","세부","네부","다섯부","여섯부","일곱부","여덟부","아홉부"
]

word_dict = {
    "주민등록":['인증로'],
    "공공" : ["00"],
    "모바일" : ["5-1","오바일","뭐 봐"],
    "전체포함" : ["전체 포함"],
    "전체미포함" : ["전체 비포함", "전체 미포함","전체비 포함"],
    "IC" : ["아이시", "아이씨"],
    "넷" : ['네','내'],
    "여섯" : ['이어서'],
    "아홉" : ['アホ'],
    "공" : ['네'],
    "삼" : ['<삼>'],
    "육" : ['이유'],
    "칠" : ['치'],
    "팔" : ['파이브'],
    "구" : ['구우','Q'],
    "네부" : ['내부'],
    "여섯부" : ['여섯 부'],
    "일곱부" : ["7부"],
    "아홉부" : ['9부'],
    "부동산" : ['공산']
}
def record_audio_with_amplification(output_filename: str, duration: int = 3, amplification_factor: float = 2.0):
    """
    실시간으로 마이크에서 오디오를 녹음하고, 증폭한 후 .wav 파일로 저장.
    
    Args:
        output_filename (str): 저장할 WAV 파일 경로.
        duration (int): 녹음 시간 (초).
        amplification_factor (float): 증폭 비율 (1.0 = 원래 크기).
    """
    # PyAudio 설정
    CHUNK = 1024  # 데이터 블록 크기
    FORMAT = pyaudio.paInt16  # 16비트 포맷
    CHANNELS = 1  # 모노 채널
    RATE = 16000  # 샘플링 레이트 (16kHz)

    p = pyaudio.PyAudio()

    # 스트림 열기
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"Recording for {duration} seconds...")

    frames = []

    # 녹음
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        audio_chunk = np.frombuffer(data, dtype=np.int16)

        # 오디오 증폭
        amplified_chunk = np.clip(audio_chunk * amplification_factor, -32768, 32767).astype(np.int16)

        frames.append(amplified_chunk.tobytes())

    print("Recording complete. Saving...")

    # 스트림 종료
    stream.stop_stream()
    stream.close()
    p.terminate()

    # WAV 파일 저장
    with wave.open(output_filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))

    print(f"File saved as {output_filename}")

record_audio_with_amplification("record.wav", duration=3, amplification_factor=2.0)
time.sleep(0.1)


def denoise_audio_with_subprocess(audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_input_file:
        temp_input_path = temp_input_file.name
        sf.write(temp_input_path, audio_data, sample_rate)

    temp_output_path = temp_input_path.replace(".wav", "_denoised.wav")

    try:
        command = ["denoise", temp_input_path, temp_output_path]
        subprocess.run(command, check=True)

        denoised_data, _ = sf.read(temp_output_path)
    finally:
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

    return denoised_data

def transcribe_audio_with_whisper(audio_data, sample_rate):
    print("Transcribing audio with Whisper...")

    temp_filename = "temp_denoised.wav"
    sf.write(temp_filename, audio_data, sample_rate)


    model = WhisperModel("medium", device="cpu", compute_type="float32")  # CPU에서 실행

    segments, info = model.transcribe(temp_filename)


    transcription = " ".join([segment.text for segment in segments])

    print("Transcription complete.")
    return transcription


def find_most_similar_word(transcription, word_list):
    # 유사도를 계산하고 가장 높은 점수를 가진 단어를 반환
    result = process.extractOne(transcription, word_list)
    
    return result  # (가장 유사한 단어, 유사도 점수)






audio_data, sample_rate = sf.read("record.wav") 


denoised_audio = denoise_audio_with_subprocess(audio_data, sample_rate)

transcription = transcribe_audio_with_whisper(denoised_audio, sample_rate)

print(f"Recognized Text: {transcription}")

result = find_most_similar_word(transcription, words_list)
print(result)

