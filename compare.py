import json
import pickle
import numpy as np
import librosa

# --- 설정 ---
JSON_ANSWER_KEY_FILE = "answer_key.json" # 1. 원곡 정답지 (1단계 산출물)
ANALYSIS_DATA_FILE = "user_analysis.pkl" # 2. 사용자 CQT 데이터 (2단계 산출물)
FEEDBACK_FILE = "feedback_report.txt"    # 3. 최종 피드백 결과물
# --------------

def time_to_cqt_frame(time_sec, sr, hop_length):
    """
    시간(초)을 CQT 프레임 인덱스로 변환합니다.
    (analyze_user.py와 동일한 로직)
    """
    return librosa.time_to_frames([time_sec], sr=sr, hop_length=hop_length)[0]

def midi_to_cqt_bin(pitch_midi, fmin_midi=36):
    """
    MIDI 피치 번호를 CQT의 '행(bin)' 인덱스로 변환합니다.
    (C2 = 36, CQT를 fmin=C2로 생성했다고 가정)
    """
    # C2(36)가 0번 행, C#2(37)가 1번 행...
    return pitch_midi - fmin_midi

# -----------------------------------------------------------------
# 메인 비교/채점 로직
# -----------------------------------------------------------------
def generate_feedback_report(answer_key_path, user_data_path):
    """
    "정답지(JSON)"와 "사용자 CQT 데이터"를 비교하여
    시간대별 피드백 리포트를 생성합니다.
    """
    
    print("--- STAGE 3: Comparing Answer Key vs User Data ---")
    
    # 1. "정답지" (JSON) 로드
    try:
        with open(answer_key_path, 'r', encoding='utf-8') as f:
            answer_notes = json.load(f)
        print(f"Loaded Answer Key: '{answer_key_path}' ({len(answer_notes)} notes)")
    except FileNotFoundError:
        print(f"ERROR: Answer Key file '{answer_key_path}' not found.")
        return

    # 2. "사용자 CQT 데이터" (Pickle) 로드
    try:
        with open(user_data_path, 'rb') as f:
            user_data = pickle.load(f)
        user_cqt_db = user_data["cqt_db"] # "열화상 사진"
        user_sr = user_data["sr"]
        user_hop_length = user_data["hop_length"] # hop_length (512)
        print(f"Loaded User Data: '{user_data_path}' (CQT Shape: {user_cqt_db.shape})")
    except FileNotFoundError:
        print(f"ERROR: User Data file '{user_data_path}' not found.")
        return
        
    feedback_report = [] # 최종 피드백을 저장할 리스트
    fmin_midi = 36 # C2 (CQT 분석 시작점)
    
    # 3. "정답지" 노트를 하나씩 순회하며 "열화상 사진"과 비교
    for i, note in enumerate(answer_notes):
        
        # (테스트용) 너무 많으니 일단 45개 노트만 피드백
        if i >= 45:
            feedback_report.append("\n... (Feedback limited to 45 notes for testing) ...")
            break
            
        t_start = note["start"]
        t_end = note["end"]
        pitch = note["pitch"]
        note_name = librosa.midi_to_note(pitch)
        
        # 4. 시간/음정 -> CQT의 (행, 열) 인덱스로 변환
        start_frame = time_to_cqt_frame(t_start, sr=user_sr, hop_length=user_hop_length)
        end_frame = time_to_cqt_frame(t_end, sr=user_sr, hop_length=user_hop_length)
        pitch_bin = midi_to_cqt_bin(pitch, fmin_midi=fmin_midi)

        # (예외 처리)
        if pitch_bin < 0 or pitch_bin >= user_cqt_db.shape[0]: continue
        if end_frame >= user_cqt_db.shape[1]: break

        # 5. "열화상 사진"에서 해당 (행, 열) 영역을 '조회'
        # [행=음높이, 열=시간]
        note_cqt_slice = user_cqt_db[pitch_bin, start_frame:end_frame]
        
        # 6. 피드백 생성 (단순 로직 예시)
        # CQT 값은 dB 단위 (0에 가까울수록 에너지가 큼)
        average_energy = np.mean(note_cqt_slice)
        
        feedback = f"[{t_start:.2f}s] Note '{note_name}': "
        
        # (임계값 예시) -25dB보다 크면 '성공' (소리가 났음)
        if average_energy > -25.0:
            feedback += f"✅ OK (Avg. Energy: {average_energy:.1f} dB)"
        else:
            # (실패) 소리가 안 났거나, 다른 음을 쳤음
            # -> 해당 시간대에 가장 크게 울린 음을 역추적
            full_slice_at_start = user_cqt_db[:, start_frame] # 시작 시점의 세로줄
            loudest_bin_index = np.argmax(full_slice_at_start)
            loudest_pitch_midi = loudest_bin_index + fmin_midi
            loudest_note_name = librosa.midi_to_note(loudest_pitch_midi)
            
            feedback += f"❌ MISSED! (Avg. Energy: {average_energy:.1f} dB). "
            feedback += f"Detected '{loudest_note_name}' instead."
            
        feedback_report.append(feedback)

    return feedback_report

# -----------------------------------------------------------------
# 메인 코드 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    report = generate_feedback_report(JSON_ANSWER_KEY_FILE, ANALYSIS_DATA_FILE)
    
    if report:
        print(f"\n\n--- 🎹 FINAL FEEDBACK REPORT (Mode 3-1) ---")
        # 1. 터미널에 출력
        for line in report:
            print(line)
            
        # 2. 파일(.txt)로 저장
        with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
            f.write("--- 🎹 FINAL FEEDBACK REPORT (Mode 3-1) ---\n")
            for line in report:
                f.write(line + "\n")
        print(f"\n✅ Feedback report saved to '{FEEDBACK_FILE}'.")