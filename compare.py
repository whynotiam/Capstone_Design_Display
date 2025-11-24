#comparison.py (연주 비교 및 피드백)
import pickle
import numpy as np
import librosa
import pretty_midi

# --- [설정] ---
ANSWER_KEY_FILE = "summer.mid"
ANALYSIS_DATA_FILE = "user_analysis.pkl"
FEEDBACK_FILE = "feedback_report.txt"

# --- [판정 파라미터] ---
TIMING_WINDOW = 1.5
TOLERANCE_PERFECT = 0.25
TOLERANCE_GOOD = 0.8

NOTE_THRESHOLD = -40.0
SILENCE_THRESHOLD = -55.0
PITCH_TOLERANCE = 1

# --- [점수 배점] ---
SCORE_PERFECT = 100
SCORE_GOOD = 95
SCORE_BAD = 60
SCORE_MISS = 0

def time_to_cqt_frame(time_sec, sr, hop_length):
    return librosa.time_to_frames([max(0, time_sec)], sr=sr, hop_length=hop_length)[0]

def midi_to_cqt_bin(pitch_midi, fmin_midi=36):
    return pitch_midi - fmin_midi

def load_notes_from_midi(midi_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes_list = []
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    notes_list.append({
                        "start": note.start,
                        "end": note.end,
                        "pitch": note.pitch
                    })
        notes_list.sort(key=lambda x: x["start"])
        return notes_list
    except Exception as e:
        print(f"ERROR: {e}")
        return []

def find_audio_start_frame(cqt_db, threshold=-50.0):
    n_bins, n_frames = cqt_db.shape
    for t in range(n_frames):
        if np.max(cqt_db[:, t]) > threshold:
            return t
    return 0

def generate_feedback_report(answer_key_path, user_data_path):
    print(f"--- STAGE 3: Final Feedback Generation ---")
    
    if answer_key_path.endswith(('.mid', '.midi')):
        answer_notes = load_notes_from_midi(answer_key_path)
    else:
        print("Error: Please use a MIDI file.")
        return

    try:
        with open(user_data_path, 'rb') as f:
            user_data = pickle.load(f)
    except FileNotFoundError:
        print("Error: User analysis file not found.")
        return
    
    user_cqt_db = user_data["cqt_db"]
    user_sr = user_data["sr"]
    user_hop_length = user_data["hop_length"]
    
    if not answer_notes: return
    midi_start_time = answer_notes[0]["start"]
    audio_start_frame = find_audio_start_frame(user_cqt_db, threshold=SILENCE_THRESHOLD)
    audio_start_time = librosa.frames_to_time(audio_start_frame, sr=user_sr, hop_length=user_hop_length)
    sync_offset = audio_start_time - midi_start_time
    
    print(f"🔍 Sync Offset Applied: {sync_offset:+.2f} sec")

    feedback_report = []
    fmin_midi = 36
    
    count_perfect = 0
    count_good = 0
    count_bad = 0
    count_miss = 0
    total_notes = 0
    
    # --- 상세 분석 루프 ---
    for i, note in enumerate(answer_notes):
        total_notes += 1
        
        target_time = note["start"] + sync_offset
        target_pitch = note["pitch"]
        target_note_name = librosa.midi_to_note(target_pitch)
        target_bin = midi_to_cqt_bin(target_pitch, fmin_midi)

        # 1. Good 범위 탐색
        good_start_time = max(0, target_time - TOLERANCE_GOOD)
        good_end_time = target_time + TOLERANCE_GOOD
        
        s_frame = time_to_cqt_frame(good_start_time, user_sr, user_hop_length)
        e_frame = time_to_cqt_frame(good_end_time, user_sr, user_hop_length)
        
        if e_frame <= s_frame: e_frame = s_frame + 1
        if e_frame > user_cqt_db.shape[1]: e_frame = user_cqt_db.shape[1]

        bin_start = max(0, target_bin - PITCH_TOLERANCE)
        bin_end = min(user_cqt_db.shape[0], target_bin + PITCH_TOLERANCE + 1)
        
        good_zone_slice = user_cqt_db[bin_start:bin_end, s_frame:e_frame]
        
        feedback = f"[Time: {note['start']:.2f}s] {target_note_name}: "
        
        if good_zone_slice.size > 0 and np.max(good_zone_slice) > NOTE_THRESHOLD:
            max_idx_flat = np.argmax(good_zone_slice)
            max_idx_2d = np.unravel_index(max_idx_flat, good_zone_slice.shape)
            played_frame = s_frame + max_idx_2d[1]
            played_time = librosa.frames_to_time(played_frame, sr=user_sr, hop_length=user_hop_length)
            
            time_diff = played_time - target_time
            abs_diff = abs(time_diff)
            timing_msg = f"(Diff: {time_diff:+.2f}s)"
            
            if abs_diff <= TOLERANCE_PERFECT:
                feedback += f"🏆 PERFECT {timing_msg}"
                count_perfect += 1
            else:
                feedback += f"🟢 GOOD {timing_msg}"
                count_good += 1
        else:
            # 2. Bad 범위 탐색
            bad_start_time = max(0, target_time - TIMING_WINDOW)
            bad_end_time = target_time + TIMING_WINDOW
            bs_frame = time_to_cqt_frame(bad_start_time, user_sr, user_hop_length)
            be_frame = time_to_cqt_frame(bad_end_time, user_sr, user_hop_length)
            if be_frame > user_cqt_db.shape[1]: be_frame = user_cqt_db.shape[1]
            
            bad_zone_slice = user_cqt_db[bin_start:bin_end, bs_frame:be_frame]
            
            if bad_zone_slice.size > 0 and np.max(bad_zone_slice) > NOTE_THRESHOLD:
                max_idx_flat = np.argmax(bad_zone_slice)
                max_idx_2d = np.unravel_index(max_idx_flat, bad_zone_slice.shape)
                played_frame = bs_frame + max_idx_2d[1]
                played_time = librosa.frames_to_time(played_frame, sr=user_sr, hop_length=user_hop_length)
                
                time_diff = played_time - target_time
                speed_msg = "Too FAST" if time_diff < 0 else "Too SLOW"
                feedback += f⚠️ BAD - {speed_msg} {timing_msg}"
                count_bad += 1
            else:
                count_miss += 1
                center_frame = time_to_cqt_frame(target_time, user_sr, user_hop_length)
                if center_frame < user_cqt_db.shape[1]:
                    full_slice = user_cqt_db[:, center_frame]
                    loudest_bin = np.argmax(full_slice)
                    if full_slice[loudest_bin] > NOTE_THRESHOLD:
                        wrong_note = librosa.midi_to_note(loudest_bin + fmin_midi)
                        feedback += f"❌ MISS (Wrong Note: {wrong_note})"
                    else:
                        feedback += f"❌ MISS (Silence)"
                else:
                    feedback += "❌ MISS"

        feedback_report.append(feedback)

    # --- 통계 계산 ---
    if total_notes > 0:
        hit_count = count_perfect + count_good + count_bad
        pitch_accuracy = (hit_count / total_notes) * 100
        
        if hit_count > 0:
            timing_accuracy = ((count_perfect + count_good) / hit_count) * 100
        else:
            timing_accuracy = 0.0
            
        weighted_score_sum = (count_perfect * SCORE_PERFECT) + \
                             (count_good * SCORE_GOOD) + \
                             (count_bad * SCORE_BAD) + \
                             (count_miss * SCORE_MISS)
        max_possible_score = total_notes * SCORE_PERFECT
        total_accuracy = (weighted_score_sum / max_possible_score) * 100
    else:
        pitch_accuracy = 0.0
        timing_accuracy = 0.0
        total_accuracy = 0.0

    # --- 결과 출력 및 저장 ---
    summary = []
    summary.append("=" * 40)
    summary.append(f"🎹 FINAL ANALYSIS REPORT")
    summary.append("=" * 40)
    summary.append(f"• Total Notes: {total_notes}")
    summary.append(f"• Perfect: {count_perfect} \t(Score: {SCORE_PERFECT})")
    summary.append(f"• Good:    {count_good} \t(Score: {SCORE_GOOD})")
    summary.append(f"• Bad:     {count_bad} \t(Score: {SCORE_BAD})")
    summary.append(f"• Miss:    {count_miss} \t(Score: {SCORE_MISS})")
    summary.append("-" * 40)
    summary.append(f"🎵 음정 정확도 (Pitch Acc):  {pitch_accuracy:.1f}%")
    summary.append(f"⏱️ 박자 정확도 (Timing Acc): {timing_accuracy:.1f}%")
    summary.append(f"⭐ 전체 정확도 (Total Score): {total_accuracy:.1f}%")
    summary.append("=" * 40)
    
    # 1. 요약 출력
    print("\n".join(summary))
    
    # 2. 상세 내용 화면 출력 
    print("\n--- Detail Feedback ---")
    # 너무 길면 터미널이 꽉 차니까 원하면 아래 숫자(50)를 조절
    # 전체를 다 보고 싶으면 [:50]을 지움
    for line in feedback_report: 
        print(line)

    # 3. 파일 저장
    with open(FEEDBACK_FILE, 'w', encoding='utf-8') as f:
        for line in summary:
            f.write(line + "\n")
        f.write("\n--- Detail Feedback ---\n")
        for line in feedback_report:
            f.write(line + "\n")
            
    print(f"\n✅ Feedback saved to '{FEEDBACK_FILE}'")
    return feedback_report

if __name__ == "__main__":
    generate_feedback_report(ANSWER_KEY_FILE, ANALYSIS_DATA_FILE)
