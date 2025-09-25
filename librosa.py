import librosa
import librosa.display
import matplotlib.pyplot as plt # 추출된 음정(f0)데이터를 시간 축에 따라 그래프로 그려줌
import numpy as np

# 1. 오디오 파일 불러오기
# 가지고 있는 노래나 악기 연주 wav 파일을 코드와 같은 폴더에 넣고 파일명을 입력
# 예: my_song.wav
audio_file = 'YOUR_AUDIO_FILE.wav' 
y, sr = librosa.load(audio_file) # 오디오 파일을 로드, y(오디오 시계열 데이터)와 sr(sampling rate, 1초당 샘플 수)로 변환

# 2. 음정(Pitch) 추출
# librosa.pyin() 함수를 사용하여 시간에 따른 기본 주파수(f0)를 추정
# f0: fundamental frequency (기본 주파수, 즉 음정)
# voiced_flag: 해당 프레임이 유성음인지(소리가 나는지) 여부 (True/False)
# voiced_probs: 유성음일 확률
# 핵심 기능 !!!!!! 오디오 데이터 (y)를 분석해 시간에 따른 기본 주파수 (f0) -> 음정이라고 부르는 값 찾아냄
# fmin, fmax는 탐색할 음정 범위 지정, 정확도 높임
f0, voiced_flag, voiced_probs = librosa.pyin(y,
                                             fmin=librosa.note_to_hz('C2'),
                                             fmax=librosa.note_to_hz('C7'))

# 3. 추출된 음정(f0)에서 값이 없는(NaN) 부분은 0으로 처리 (시각화를 위해)
f0[np.isnan(f0)] = 0

# 4. 시간 축 생성
times = librosa.times_like(f0)

# 5. 결과 시각화
fig, ax = plt.subplots()
# D: 점선 스타일로 파형을 그림
librosa.display.waveshow(y, sr=sr, alpha=0.5, ax=ax)
ax.plot(times, f0, label='f0', color='cyan', linewidth=3)
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Pitch Contour')
ax.legend()
plt.show()

# 6. (보너스) 추출된 주파수 값을 음악 노트 이름으로 변환하여 출력
# 첫 20개의 유성음 구간에 대해서만 노트 이름 출력
count = 0
for i, freq in enumerate(f0):
    if voiced_flag[i] and count < 20:
        note_name = librosa.hz_to_note(freq) # 추출된 주파수 값을 음악적 노트 이름으로 변환
        print(f"Time: {times[i]:.2f}s, Frequency: {freq:.2f}Hz, Note: {note_name}")
        count += 1