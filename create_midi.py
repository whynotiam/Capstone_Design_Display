import pretty_midi
import music21
import os
import json

# [수정] basic-pitch 라이브러리 및 "기본 모델" 불러오기
from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH 

# --- 설정 ---
ORIGINAL_MP3_FILE = "original_piano_solo.mp3" 
TEMP_MIDI_FILE = "temp_original.mid" 
JSON_ANSWER_KEY_FILE = "answer_key.json"
MXL_SHEET_MUSIC_FILE = "sheet_music.mxl"
# --------------

def create_answer_key_from_mp3_basic_pitch(mp3_path):
    """
    [개선됨] 'basic-pitch'를 사용하여 MP3를 분석하고 정답지를 생성합니다.
    """
    print(f"--- STAGE 1: Analyzing Original Song (basic-pitch) ---")
    print(f"Input file: {mp3_path}")

    # --- A. MP3 -> MIDI (by basic-pitch) ---
    print("Starting basic-pitch transcription... (This may take a while)")
    
    output_directory = "." # 현재 폴더
    
    try:
        # 'predict_and_save' 함수의 "올바른" 인자 순서로 호출
        predict_and_save(
            [mp3_path],                   # 1. MP3 경로 리스트
            output_directory,             # 2. 출력 폴더
            save_midi=True,
            sonify_midi=False,
            save_model_outputs=False,
            save_notes=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH # 모델은 맨 뒤에 옵션으로 지정
        )

        # basic-pitch는 파일명 뒤에 _basic_pitch.mid를 붙입니다.
        generated_midi_path = mp3_path.replace(".mp3", "_basic_pitch.mid")
        
        if not os.path.exists(generated_midi_path):
            print(f"ERROR: basic-pitch did not create the file '{generated_midi_path}'.")
            return False
            
        os.rename(generated_midi_path, TEMP_MIDI_FILE)
        
        print("basic-pitch transcription complete -> Temporary MIDI file created.")

    except Exception as e:
        print(f"FATAL ERROR during basic-pitch transcription (Part A): {e}")
        return False

    # --- B. MIDI -> JSON (by pretty_midi) ---
    print("Converting MIDI to 'JSON Answer Key'... (for comparison)")
    try:
        midi_data = pretty_midi.PrettyMIDI(TEMP_MIDI_FILE)
        note_list = []
        
        for instrument in midi_data.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    note_list.append({
                        "pitch": note.pitch,
                        "start": note.start,
                        "end": note.end
                    })
                break
        
        with open(JSON_ANSWER_KEY_FILE, 'w', encoding='utf-8') as f:
            json.dump(note_list, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 'JSON Answer Key' created successfully: {JSON_ANSWER_KEY_FILE} ({len(note_list)} notes)")

    except Exception as e:
        print(f"ERROR during JSON conversion (Part B): {e}")
        # [수정] Part B가 실패하면 여기서 즉시 중단 (MIDI 파일 삭제 안 함)
        print(f"NOTE: '{TEMP_MIDI_FILE}' file is kept for debugging.")
        return False

    
# -----------------------------------------------------------------
# 메인 코드 실행
# -----------------------------------------------------------------
if __name__ == "__main__":
    if not os.path.exists(ORIGINAL_MP3_FILE):
        print(f"ERROR: '{ORIGINAL_MP3_FILE}' not found.")
        print("Please put 'original_piano_solo.mp3' in the same folder as this script.")
    else:
        create_answer_key_from_mp3_basic_pitch(ORIGINAL_MP3_FILE)