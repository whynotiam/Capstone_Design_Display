# convert_to_sheet.py
# (BasicPitch MIDI의 '유령 이벤트' 버그를 수정한 최종본)

import os
from copy import deepcopy
import numpy as np
import pretty_midi

from music21 import converter, stream, note, chord, clef, instrument, meter, layout, tempo, environment

MIDI_FILE_PATH    = "summerhigh.mid"
CLEAN_MIDI_PATH   = "song_clean.mid"
OUTPUT_XML_PATH   = "song_sheet.musicxml"
TARGET_DURATION_S = 180.0   # 2분 14초
SPLIT_POINT_MIDI  = 60      # C4 기준
SUBDIV_PER_BEAT   = 4       # 16분음표 스냅

# MuseScore 경로(없으면 show는 패스)
try:
    environment.UserSettings()['musicxmlPath'] = r'C:\Users\junga\AppData\Local\Programs\MuseScore 4\bin\MuseScore4.exe'
except Exception:
    print("경고: MuseScore 경로를 찾지 못했습니다. .show()가 작동하지 않을 수 있습니다.")
    pass

def snap_times_to_grid(pm: pretty_midi.PrettyMIDI, subdiv=4):
    """템포 맵 기반 그리드 생성 후 start/end를 가장 가까운 그리드로 스냅."""
    beat_times = pm.get_beats()
    if len(beat_times) < 2:
        total = pm.get_end_time()
        grid = np.arange(0.0, total + 1e-6, 0.5 / subdiv) # 120bpm 가정
    else:
        pieces = []
        for i in range(len(beat_times) - 1):
            t0, t1 = beat_times[i], beat_times[i+1]
            seg = np.linspace(t0, t1, subdiv+1, endpoint=False)
            pieces.append(seg)
        grid = np.concatenate(pieces + [np.array([beat_times[-1]])])

    def snap(t):
        idx = np.searchsorted(grid, t)
        if idx == 0:
            return grid[0]
        if idx >= len(grid):
            return grid[-1]
        return grid[idx] if abs(grid[idx]-t) <= abs(t-grid[idx-1]) else grid[idx-1]

    for inst in pm.instruments:
        for n in inst.notes:
            s = snap(n.start)
            e = snap(n.end)
            if e <= s: 
                e = s + max(0.03, (grid[1]-grid[0]) if len(grid) > 1 else 0.03)
            n.start, n.end = s, e

def preprocess_midi(in_path: str, out_path: str, target_len_s: float):
    """
    (이 함수는 훌륭합니다. 그대로 유지합니다)
    """
    pm = pretty_midi.PrettyMIDI(in_path)

    # 템포 없으면 120bpm 마커 하나 추가
    times, tempi = pm.get_tempo_changes()
    if len(times) == 1 and tempi[0] == 120.0:
        pass 
    if pm.get_end_time() == 0:
        raise RuntimeError("Empty MIDI")

    # sustain(CC64) 반영
    for inst in pm.instruments:
        on = None
        for cc in inst.control_changes:
            if cc.number != 64:
                continue
            if cc.value >= 64 and on is None:
                on = cc.time
            elif cc.value < 64 and on is not None:
                off = cc.time
                for n in inst.notes:
                    if (n.start < off) and (n.end > on) and n.end < off:
                        n.end = off
                on = None
    
    # [수정됨] 유령 이벤트를 막기 위해 CC도 컷오프
    cutoff = target_len_s + 0.30
    for inst in pm.instruments:
        inst.control_changes = [cc for cc in inst.control_changes if cc.time < cutoff]
    # --- [수정 끝] ---

    # 약한 필터만 적용
    VELOCITY_MIN = 8 
    MIN_DUR      = 0.025

    for inst in pm.instruments:
        if not inst.notes:
            continue
        kept = []
        for n in inst.notes:
            if n.start >= cutoff:
                continue
            if n.end > cutoff:
                n.end = cutoff
            if n.velocity < VELOCITY_MIN:
                continue
            if (n.end - n.start) < MIN_DUR:
                continue
            kept.append(n)
        inst.notes = kept

        # 같은 피치가 15ms 이내로 붙어 있으면 연결
        inst.notes.sort(key=lambda x: (x.pitch, x.start))
        merged = []
        for n in inst.notes:
            if merged and merged[-1].pitch == n.pitch and (n.start - merged[-1].end) <= 0.015:
                merged[-1].end = max(merged[-1].end, n.end)
                merged[-1].velocity = max(merged[-1].velocity, n.velocity)
            else:
                merged.append(n)
        inst.notes = merged

    # 그리드 스냅(16분음표 기준)
    snap_times_to_grid(pm, subdiv=SUBDIV_PER_BEAT)

    # 마지막 안전 트림
    end_time = pm.get_end_time()
    if end_time > cutoff:
        for inst in pm.instruments:
            inst.notes = [n for n in inst.notes if n.start < cutoff]
            for n in inst.notes:
                if n.end > cutoff:
                    n.end = cutoff
    
    pm.write(out_path)

def build_score_from_midi(mid_path: str):
    """
    [최종 수정]
    music21의 .quantize()와 .chordify() 마법을 사용하여
    '인간적인' MIDI를 '악보용' 화음/박자로 변환합니다.
    """
    s = stream.Score(id='score')
    right = stream.Part(id='RH'); right.insert(0, instrument.Piano()); right.insert(0, clef.TrebleClef())
    left  = stream.Part(id='LH'); left.insert(0, instrument.Piano());  left.insert(0, clef.BassClef())

    # 1. "인간적인" MIDI 로드 (수정 X)
    midi = converter.parse(mid_path)

    # --- [!!!] "독"을 제거하고 "마법"을 부리는 곳 [!!!] ---
    print("Applying music21 'quantize' and 'chordify' magic...")
    
    # 2. [핵심 마법 1] 양자화(Quantize)
    #    "인간적인" 박자를 "악보" 박자로 스냅 (16분음표 기준)
    #    이것이 'snap_times_to_grid'를 대체하는 "전문가" 버전입니다.
    midi_quantized = midi.quantize([SUBDIV_PER_BEAT]) # [SUBDIV_PER_BEAT] = 4

    # 3. [핵심 마법 2] 화음화(Chordify)
    #    동시에 울리는(양자화된) 'note.Note'들을 'chord.Chord'로 묶음
    midi_chordified = midi_quantized.chordify()
    
    # ----------------------------------------------------

    # 4. "마법이 적용된" MIDI에서 박자/템포 추출
    ts = midi_chordified.recurse().getElementsByClass('TimeSignature').first() or meter.TimeSignature('4/4')
    mm = midi_chordified.recurse().getElementsByClass('MetronomeMark').first() or tempo.MetronomeMark(number=120)

    for tgt in (right, left):
        tgt.insert(0, deepcopy(ts))
        tgt.insert(0, deepcopy(mm))

    # 5. [수정] "마법이 적용된" (midi_chordified) 스트림을 순회
    #    이제 el은 'note.Note'가 아니라 'chord.Chord'가 됩니다.
    for el in midi_chordified.flat.notes: # [수정] midi -> midi_chordified
        off = float(el.offset)
        ql  = float(el.duration.quarterLength)

        if isinstance(el, note.Note):
            # (이제 단음은 이쪽으로)
            tgt = right if el.pitch.midi >= SPLIT_POINT_MIDI else left
            n = deepcopy(el); n.duration.quarterLength = ql
            tgt.insert(off, n)

        elif isinstance(el, chord.Chord):
            # (이제 90%의 음은 이쪽으로 들어옴!)
            treble = [p for p in el.pitches if p.midi >= SPLIT_POINT_MIDI]
            bass   = [p for p in el.pitches if p.midi <  SPLIT_POINT_MIDI]
            if treble:
                right.insert(off, chord.Chord(treble, quarterLength=ql))
            if bass:
                left.insert(off,  chord.Chord(bass,   quarterLength=ql))

    # (이하 코드는 동일)
    right.makeMeasures(inPlace=True); left.makeMeasures(inPlace=True)
    s.insert(0, right); s.insert(0, left)
    s.insert(0, layout.StaffGroup([right, left], name='Piano', symbol='brace'))
    try:
        s.makeNotation(inPlace=True)
    except Exception:
        pass
    return s
    
    
if __name__ == "__main__":
    if not os.path.exists(MIDI_FILE_PATH):
        print(f"오류: '{MIDI_FILE_PATH}'가 없습니다.")
        raise SystemExit(1)

    print("1) 약한 전처리 + 하드 트림 + 스냅...")
    preprocess_midi(MIDI_FILE_PATH, CLEAN_MIDI_PATH, TARGET_DURATION_S)

    print("2) 악보 생성...")
    score = build_score_from_midi(CLEAN_MIDI_PATH)

    print("3) MusicXML 저장...")
    score.write('musicxml', fp=OUTPUT_XML_PATH)
    print(f"완료: {OUTPUT_XML_PATH}")

    try:
        print("4) 뷰어(MuseScore) 실행 시도...")
        score.show('musicxml')
    except Exception as e:
        print(f"자동 표시 실패: {e}\nMuseScore에서 직접 여세요.")