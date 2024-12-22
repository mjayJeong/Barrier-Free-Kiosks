import os
import tempfile
import subprocess
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel
from rapidfuzz import process
import pyaudio
import wave




class VoiceTranscriber:
    def __init__(self, mode=1):
        """
        VoiceTranscriber 초기화.
        mode: 현재 예제에서는 사용하지 않으나, 확장 가능성을 위해 포함.
        """
        self.mode = mode  # mode는 추가 옵션으로 사용 가능
        self.words_list = ["모바일","등본","확인","법원","교육","공공","부동산","금융","병원",
              "전체포함","전체미포함","예","아니","카드","현금","IC","주민등록","발급",
              "영","하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉",
              "공","일","이","삼","사","오","육","칠","팔","구",
              "한부","두부","세부","네부","다섯부","여섯부","일곱부","여덟부","아홉부"]
        self.word_dict = {
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

    def record_audio(self, duration=3, amplification_factor=2.0):
        """
        실시간으로 오디오를 녹음하고 증폭한 후 반환.
        """
        import pyaudio
        import wave

        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()

        # 마이크 스트림 열기
        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print(f"Recording for {duration} seconds...")

        frames = []

        # 녹음 루프
        for _ in range(0, int(RATE / CHUNK * duration)):
            data = stream.read(CHUNK)
            audio_chunk = np.frombuffer(data, dtype=np.int16)

            # 오디오 증폭
            amplified_chunk = np.clip(audio_chunk * amplification_factor, -32768, 32767).astype(np.int16)
            frames.append(amplified_chunk.tobytes())

        print("Recording complete.")

        # 스트림 종료
        stream.stop_stream()
        stream.close()
        p.terminate()

        # WAV 파일로 저장
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            output_path = temp_file.name
            with wave.open(output_path, 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(p.get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b''.join(frames))

        print(f"Temporary audio file saved at {output_path}")
        return output_path

    def denoise_audio(self, input_path):
        """
        소음 제거를 수행한 오디오 데이터를 반환.
        """
        output_path = input_path.replace(".wav", "_denoised.wav")

        try:
            command = ["denoise", input_path, output_path]
            subprocess.run(command, check=True)
            print("Denoising complete.")
        except Exception as e:
            print(f"Error during denoising: {e}")
            return input_path

        return output_path

    def transcribe(self):
        """
        음성을 녹음하고 Whisper로 텍스트를 반환.
        """
        recorded_audio_path = self.record_audio()
        denoised_audio_path = self.denoise_audio(recorded_audio_path)

        print("Transcribing audio with Whisper...")

        model = WhisperModel("medium", device="cpu", compute_type="float32")  # CPU에서 실행
        segments, _ = model.transcribe(denoised_audio_path,language ='ko')

        transcription = " ".join([segment.text for segment in segments])

        print("Transcription complete.")
        print(f"Recognized Text: {transcription}")

        # 임시 파일 삭제
        os.remove(recorded_audio_path)
        if os.path.exists(denoised_audio_path):
            os.remove(denoised_audio_path)

        result = process.extractOne(transcription, self.words_list)
        print(result,"result")
        return result[0]



def get_transcriber(mode=1):
    """
    VoiceTranscriber 객체를 반환.
    """
    return VoiceTranscriber(mode)
