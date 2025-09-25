# 후에 'pitch_analysis.py' 파일과 합치는 작업 할 것

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# 1. 오디오 파일 불러오기 (WAV, 기본 sr)
audio_file = 'YOUR_AUDIO_FILE.wav'
y, sr = librosa.load(audio_file)

# 2. 온셋(Onset) 탐지
# 소리가 시작되는 지점(프레임 번호) 찾기
# 핵심 기능 !! 오디오 신호의 에너지가 급격히 변하는 지점 감지 -> 소리 시작 판단
onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
# 찾은 프레임 번호를 시간(초)으로 변환
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# 3. 박자(Beat) 및 템포(Tempo) 추적
# 음악의 전체적인 박자를 추적하여 템포(BPM)와 박자가 찍히는 프레임 획득
# 핵심 기능 2 !! 온셋 정보를 포함한 다양한 특징 종합 -> 음악 전반이 템포와 규칙적 박자 위치 찾음
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
# 박자 프레임도 시간(초)으로 변환
# librosa의 분석 결과는 대부분 프레임 번호로 나오기 때문에 우리가 이해하기 쉬운 초 단위로 변환
beat_times = librosa.frames_to_time(beat_frames, sr=sr)

# 4. 결과 출력
print(f"Estimated Tempo: {tempo:.2f} BPM")

# 첫 10개의 온셋 시간만 출력
print("\n--- Onset Times (first 10) ---")
for t in onset_times[:10]:
    print(f"{t:.3f} s")
    
# 5. 결과 시각화
fig, ax = plt.subplots(figsize=(15, 5))

# 오디오 파형 표시
librosa.display.waveshow(y, sr=sr, alpha=0.6, ax=ax, label='Waveform')

# ax.vlines(): Matplotlib 함수. 지정된 시간에 수직선 그림
# 탐지된 온셋 위치에 빨간색 점선으로 수직선 그리기
ax.vlines(onset_times, ymin=-1, ymax=1, color='r', linestyle='--', label='Onsets')

# 탐지된 박자 위치에 초록색 실선으로 수직선 그리기
ax.vlines(beat_times, ymin=-1, ymax=1, color='g', linestyle='-', label='Beats')

ax.set_title(f'Onset and Beat Detection (Tempo: {tempo:.2f} BPM)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.legend()
plt.show()