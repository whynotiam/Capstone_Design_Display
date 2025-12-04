import pickle
import numpy as np
import librosa
import pretty_midi
import os

# --- [설정] ---
ANSWER_KEY_FILE = "summerrightmiss.mid" 
ANALYSIS_DATA_FILE = "user_analysis.pkl" 
FEEDBACK_FILE = "feedback_report.txt"

# --- [분석 상수] ---
PITCH_RESOLUTION_FACTOR = 2 

# --- [판정 기준: 더 유연하게] ---
TIMING_WINDOW = 3.0       # 탐색 범위를 3초로 늘림 (싱크 오차 커버)
TOLERANCE_PERFECT = 0.5   
TOLERANCE_GOOD = 1.5      # 1.5초까지 봐줌

# 소리 감지 임계값 (조금 더 낮춰서 작은 소리도 잡음)
NOTE_THRESHOLD = -50.0    

# 주파수 허용 범위 (반음 3~4개 차이까지 봐줌 -> 배음 문제 해결)
PITCH_TOLERANCE = 4       

# --- [점수] ---
SCORE_PERFECT = 100
SCORE_GOOD = 90
SCORE_BAD = 50
SCORE_MISS = 0

# ---------------------------------------------------------
# 헬퍼 함수
# ---------------------------------------------------------
def time_to_cqt_frame(time_sec, sr, hop_length):
    return librosa.time_to_frames([max(0, time_sec)], sr=sr, hop_length=hop_length)[0]

def midi_to_cqt_bin(pitch_midi, fmin_midi=36):
    return (pitch_midi - fmin_midi) * PITCH_RESOLUTION_FACTOR

def load_notes_from_midi(midi_path):
    if not os.path.exists(midi_path): return None
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes_list = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes_list.append({
                        "start": note.start, "end": note.end, "pitch": note.pitch
                    })
        notes_list.sort(key=lambda x: x["start"])
        return notes_list
    except: return None

def find_first_musical_onset(cqt_db, sr, hop_length):
    S = librosa.db_to_amplitude(cqt_db)
    onset_env = librosa.onset.onset_strength(S=S, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop_length, backtrack=False)
    for frame in onset_frames:
        if np.max(cqt_db[:, frame]) > NOTE_THRESHOLD:
            return frame
    return 0

# ---------------------------------------------------------
# 메인 분석 로직
# ---------------------------------------------------------
def generate_feedback_report(answer_path, user_path):
    print(f"--- STAGE 3: Enhanced Tolerance Analysis ---")
    
    answer_notes = load_notes_from_midi(answer_path)
    if answer_notes is None: return

    try:
        with open(user_path, 'rb') as f:
            user_data = pickle.load(f)
    except: return

    user_cqt_db = user_data["cqt_db"]
    user_sr = user_data["sr"]
    user_hop_length = user_data["hop_length"]
    
    # 1. 싱크 맞추기
    midi_start_time = answer_notes[0]["start"]
    audio_start_frame = find_first_musical_onset(user_cqt_db, user_sr, user_hop_length)
    audio_start_time = librosa.frames_to_time(audio_start_frame, sr=user_sr, hop_length=user_hop_length)
    sync_offset = audio_start_time - midi_start_time
    
    raw_feedbacks = [] 
    fmin_midi = 36
    
    count_perfect = 0
    count_good = 0
    count_bad = 0
    count_miss = 0
    
    for i, note in enumerate(answer_notes):
        # 시작 1초 무시
        if note["start"] < 1.0:
            continue

        target_time = note["start"] + sync_offset
        target_pitch = note["pitch"]
        target_note_name = librosa.midi_to_note(target_pitch)
        
        s_frame = time_to_cqt_frame(max(0, target_time - TOLERANCE_GOOD), user_sr, user_hop_length)
        e_frame = time_to_cqt_frame(target_time + TOLERANCE_GOOD, user_sr, user_hop_length)
        if e_frame > user_cqt_db.shape[1]: e_frame = user_cqt_db.shape[1]
        
        full_slice = user_cqt_db[:, s_frame:e_frame]
        
        status = "MISS"
        msg = "MISS (Silence)"
        is_pitch_correct = False
        
        if full_slice.size > 0:
            # [핵심 수정] 가장 큰 소리 하나만 찾는 게 아니라, 정답 근처의 소리를 먼저 찾음
            # 정답 Bin 계산
            target_bin = midi_to_cqt_bin(target_pitch, fmin_midi)
            
            # 정답 Bin 주변(PITCH_TOLERANCE) 슬라이싱
            bin_start = int(max(0, target_bin - PITCH_TOLERANCE))
            bin_end = int(min(user_cqt_db.shape[0], target_bin + PITCH_TOLERANCE + 1))
            
            target_area = full_slice[bin_start:bin_end, :]
            
            # 정답 영역에서 소리가 임계값보다 크면 맞았다고 인정!
            if target_area.size > 0 and np.max(target_area) > NOTE_THRESHOLD:
                is_pitch_correct = True
                
                # 타이밍 계산 (정답 영역 내에서 가장 큰 지점)
                local_max = np.argmax(target_area)
                local_idx = np.unravel_index(local_max, target_area.shape)
                played_frame = s_frame + local_idx[1]
                played_time = librosa.frames_to_time(played_frame, sr=user_sr, hop_length=user_hop_length)
                
            else:
                # 정답 영역에 소리가 없음 -> 엉뚱한 음을 쳤는지 확인
                is_pitch_correct = False
        
        # --- 판정 결과 생성 ---
        if is_pitch_correct:
            time_diff = played_time - target_time
            abs_diff = abs(time_diff)
            
            timing_txt = ""
            if time_diff < -0.05: timing_txt = f"Fast {abs_diff:.2f}s"
            elif time_diff > 0.05: timing_txt = f"Slow {abs_diff:.2f}s"
            else: timing_txt = "Perfect"

            if abs_diff <= TOLERANCE_PERFECT:
                status = "PERFECT"
                msg = "PERFECT"
                count_perfect += 1
            elif abs_diff <= TOLERANCE_GOOD:
                status = "GOOD"
                msg = f"GOOD ({timing_txt})"
                count_good += 1
            else:
                # 0.5초 이상 차이나면 BAD
                status = "BAD"
                msg = f"BAD ({timing_txt})"
                count_bad += 1
        else:
            # 피치가 틀렸거나 소리가 안 남
            # 전체 영역에서 가장 큰 소리가 있었는지 확인 (Wrong Note 판별)
            if full_slice.size > 0 and np.max(full_slice) > NOTE_THRESHOLD:
                loudest_idx = np.unravel_index(np.argmax(full_slice), full_slice.shape)
                wrong_bin = loudest_idx[0]
                wrong_pitch = (wrong_bin / PITCH_RESOLUTION_FACTOR) + fmin_midi
                wrong_note = librosa.midi_to_note(int(round(wrong_pitch)))
                
                status = "WRONG"
                # 틀린 음을 쳤어도 박자는 대략 계산해서 알려줌
                played_frame = s_frame + loudest_idx[1]
                played_time = librosa.frames_to_time(played_frame, sr=user_sr, hop_length=user_hop_length)
                time_diff = played_time - target_time
                timing_txt = f"Fast {abs(time_diff):.2f}s" if time_diff < 0 else f"Slow {abs(time_diff):.2f}s"
                
                msg = f"WRONG NOTE (Played: {wrong_note}, Rhythm: {timing_txt})"
                count_bad += 1
            else:
                status = "MISS"
                msg = "SILENCE"
                count_miss += 1

        raw_feedbacks.append({
            "time": note["start"], 
            "status": status,
            "msg": msg
        })

    # 피드백 리포트 생성 (Perfect 제외)
    grouped_feedbacks = [f for f in raw_feedbacks if f["status"] != "PERFECT" and f["status"] != "GOOD"]

    total_notes = len(answer_notes)
    if total_notes > 0:
        score = (count_perfect * SCORE_PERFECT) + (count_good * SCORE_GOOD) + \
                (count_bad * SCORE_BAD) + (count_miss * SCORE_MISS)
        total_acc = (score / (total_notes * SCORE_PERFECT)) * 100
        
        hit_count = count_perfect + count_good + count_bad
        pitch_acc = (hit_count / total_notes) * 100
        timing_acc = ((count_perfect + count_good) / hit_count * 100) if hit_count > 0 else 0
    else:
        total_acc = 0
        pitch_acc = 0
        timing_acc = 0

    summary = [f"종합 점수: {total_acc:.1f}점"]
    
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        for line in summary: f.write(line + "\n")
        f.write("\n--- Detail Feedback ---\n")
        for item in grouped_feedbacks:
            f.write(f"[Time: {item['time']:.2f}s] {item['msg']}\n")
            
    print(f"\n✅ Feedback saved to '{FEEDBACK_FILE}'")

if __name__ == "__main__":
    generate_feedback_report(ANSWER_KEY_FILE, ANALYSIS_DATA_FILE)