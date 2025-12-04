import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

# --- 설정 ---
USER_MP3_FILE = "summerrightper.mp3" 
ANALYSIS_DATA_FILE = "user_analysis.pkl" 
CQT_HOP_LENGTH = 512 
# -----------------

# [NEW CONSTANT] PITCH 해상도 2배 증가
PITCH_RESOLUTION_FACTOR = 2 
N_BINS_HIGH_RES = 72 * PITCH_RESOLUTION_FACTOR       # 144 bins
BINS_PER_OCTAVE_HIGH_RES = 12 * PITCH_RESOLUTION_FACTOR # 24 bins

def analyze_user_performance_librosa(mp3_path: str) -> dict:
    """
    [고해상도 CQT 적용] 왼손 연주 정확도를 높입니다.
    """
    
    print(f"--- STAGE 2: Analyzing User Performance (High-Res CQT) ---")
    print(f"Input file: {mp3_path}")
    
    try:
        y, sr = librosa.load(mp3_path, sr=None)
        print(f"Audio loaded (sr={sr}, duration={librosa.get_duration(y=y, sr=sr):.2f}s)")
    except Exception as e:
        print(f"ERROR: Could not load file '{mp3_path}': {e}")
        return None

    # 2. CQT (다선율 음정)
    print(f"Generating CQT spectrogram (Resolution: {BINS_PER_OCTAVE_HIGH_RES} Bins/Octave)...")
    
    C = librosa.cqt(y, sr=sr, 
                    fmin=librosa.note_to_hz('C2'), 
                    n_bins=N_BINS_HIGH_RES, 
                    bins_per_octave=BINS_PER_OCTAVE_HIGH_RES,
                    hop_length=CQT_HOP_LENGTH) 
    C_db = librosa.amplitude_to_db(np.abs(C), ref=np.max)

    # 3. 박자(Onset) 
    print("Extracting Onsets (Rhythm)...")
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=CQT_HOP_LENGTH)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=CQT_HOP_LENGTH)

    print(f"Detected {len(onset_times)} note onsets. CQT Shape: {C_db.shape}")

    # 4. 결과 정리 (🚨 에러 발생 라인 클린업 🚨)
    analysis_result = {
        "cqt_db": C_db,
        "onset_frames": onset_frames,
        "onset_times": onset_times,
        "sr": sr,
        "hop_length": CQT_HOP_LENGTH, 
        "bins_per_octave": BINS_PER_OCTAVE_HIGH_RES 
    }
    
    print(f"✅ User analysis complete.")
    return analysis_result

def visualize_user_analysis(result: dict):
    """
    Librosa CQT와 Onset 분석 결과를 시각화합니다. (해상도 자동 적용)
    """
    print("Visualizing analysis...")
    
    sr = result["sr"]
    C_db = result["cqt_db"]
    onset_times = result["onset_times"]
    
    fig, ax = plt.subplots(figsize=(15, 6))
    
    librosa.display.specshow(C_db, sr=sr,
                             x_axis='time', y_axis='cqt_note',
                             fmin=librosa.note_to_hz('C2'), 
                             bins_per_octave=result["bins_per_octave"], 
                             hop_length=result["hop_length"], ax=ax)
    
    # VLines Y Max/Min 조정
    y_max_hz = librosa.note_to_hz(f'C{2 + C_db.shape[0] // result["bins_per_octave"]}')
    
    ax.vlines(onset_times,
              ymin=librosa.note_to_hz('C2'), 
              ymax=y_max_hz,
              color='r', linestyle='--', alpha=0.8, label='Onsets')

    ax.set_title(f'User Performance Analysis (High Resolution CQT)')
    ax.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------
# 메인 코드 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    
    if not os.path.exists(USER_MP3_FILE):
        print(f"ERROR: '{USER_MP3_FILE}' not found.")
    else:
        user_data = analyze_user_performance_librosa(USER_MP3_FILE)
        
        if user_data:
            # visualize_user_analysis(user_data) # 시각화는 주석 처리 (선택 사항)
            
            with open(ANALYSIS_DATA_FILE, 'wb') as f:
                pickle.dump(user_data, f) 
            print(f"Analysis results saved to '{ANALYSIS_DATA_FILE}'.")