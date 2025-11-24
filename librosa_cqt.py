import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# --- 설정 ---
USER_MP3_FILE = "user_performance.mp3" 
ANALYSIS_DATA_FILE = "user_analysis.pkl" 
CQT_HOP_LENGTH = 512 # CQT/Onset의 시간 단위를 통일 (핵심)
# -----------------

def analyze_user_performance_librosa(mp3_path: str) -> dict:
    """
    [수정됨] CQT와 Onset의 hop_length를 512로 통일합니다.
    """
    
    print(f"--- STAGE 2: Analyzing User Performance (Librosa CQT) ---")
    print(f"Input file: {mp3_path}")
    
    try:
        y, sr = librosa.load(mp3_path, sr=None)
        print(f"Audio loaded (sr={sr}, duration={librosa.get_duration(y=y, sr=sr):.2f}s)")
    except Exception as e:
        print(f"ERROR: Could not load file '{mp3_path}': {e}")
        return None

    # 2. CQT (다선율 음정)
    print("Generating CQT spectrogram (Pitch)...")
    # [수정] hop_length를 512로 명시
    C = librosa.cqt(y, sr=sr, 
                    fmin=librosa.note_to_hz('C2'), 
                    n_bins=72,
                    hop_length=CQT_HOP_LENGTH) 
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    # 3. 박자(Onset) (페달 없음 가정)
    print("Extracting Onsets (Rhythm)...")
    # [수정] hop_length를 512로 명시
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=CQT_HOP_LENGTH)
    
    # [수정] frames_to_time에도 hop_length 적용
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=CQT_HOP_LENGTH)

    print(f"Detected {len(onset_times)} note onsets.")

    # 4. 결과 정리
    analysis_result = {
        "cqt_db": C_db,               # CQT 데이터 (음정 히트맵)
        "onset_frames": onset_frames, # CQT 프레임과 동일한 기준의 박자
        "onset_times": onset_times,   # 초 단위 박자
        "sr": sr,
        "hop_length": CQT_HOP_LENGTH  # [추가] 비교 코드(3단계)가 참조할 값
    }
    
    print(f"✅ User analysis complete.")
    return analysis_result

def visualize_user_analysis(result: dict):
    """
    Librosa CQT와 Onset 분석 결과를 시각화합니다.
    """
    print("Visualizing analysis...")
    
    sr = result["sr"]
    C_db = result["cqt_db"]
    onset_times = result["onset_times"] # 시각화는 초 단위 Onset 사용

    fig, ax = plt.subplots(figsize=(15, 6))
    
    # CQT 히트맵 그리기
    librosa.display.specshow(C_db, sr=sr,
                             x_axis='time', y_axis='cqt_note',
                             fmin=librosa.note_to_hz('C2'), 
                             # n_bins=72,  # <-- [수정] 이 줄을 삭제!
                             hop_length=result["hop_length"], ax=ax)
    
    ax.vlines(onset_times,
              ymin=librosa.note_to_hz('C2'), 
              ymax=librosa.note_to_hz(f'C{2 + 72//12}'),
              color='r', linestyle='--', label='Onsets')

    ax.set_title('User Performance Analysis (CQT & Onset)')
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# 메인 코드 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    if not os.path.exists(USER_MP3_FILE):
        print(f"ERROR: '{USER_MP3_FILE}' not found.")
        print("Please put 'user_performance.mp3' in the same folder.")
    else:
        user_data = analyze_user_performance_librosa(USER_MP3_FILE)
        
        if user_data:
            visualize_user_analysis(user_data)
            
            with open(ANALYSIS_DATA_FILE, 'wb') as f:
                pickle.dump(user_data, f)
            print(f"Analysis results saved to '{ANALYSIS_DATA_FILE}'.")