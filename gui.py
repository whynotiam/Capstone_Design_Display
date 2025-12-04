import sys
import os
import pickle
import re

import PyQt5
pyqt_path = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(pyqt_path, 'Qt5', 'plugins')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path
print(f"🔌 플러그인 경로 강제 설정 완료: {plugin_path}")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget, QFileDialog, 
                             QMessageBox, QPushButton, QLabel, QWidget)
from PyQt5.QtCore import QTimer, Qt, QThread, pyqtSignal
from PyQt5 import uic

try:
    import analysis
    import comparison
    import librosa
except ImportError as e:
    print(f"[Critical Error] 필수 모듈이 없습니다: {e}")
    sys.exit()

os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
os.environ["QT_SCALE_FACTOR"] = "1"

HISTORY_FILE = "midi_history.pkl"

def seconds_to_min_sec(seconds):
    if seconds is None: seconds = 0
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

class AnalysisWorker(QThread):
    finished_signal = pyqtSignal(dict)

    def __init__(self, midi_path, mp3_path):
        super().__init__()
        self.midi_path = midi_path
        self.mp3_path = mp3_path
        self.pkl_path = "user_analysis.pkl"
        self.report_path = "feedback_report.txt"

    def run(self):
        result_data = {"accuracy": 0, "duration": 0, "feedback_list": []}
        try:
            print("--- Analysis Start ---")
            user_data = analysis.analyze_user_performance_librosa(self.mp3_path)
            if user_data is None: raise Exception("MP3 분석 실패")

            duration = librosa.get_duration(sr=user_data['sr'], S=user_data['cqt_db'], hop_length=user_data['hop_length'])
            result_data["duration"] = duration

            with open(self.pkl_path, 'wb') as f:
                pickle.dump(user_data, f)

            comparison.generate_feedback_report(self.midi_path, self.pkl_path)
            self.parse_feedback_report(result_data)

        except Exception as e:
            print(f"[Worker Error] {e}")
        
        self.finished_signal.emit(result_data)

    def parse_feedback_report(self, result_data):
        if not os.path.exists(self.report_path): return
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        feedback_list = []
        line_pattern = re.compile(r"\[Time:\s*(\d+\.\d+)s\]\s*(.*)")
        score_pattern = re.compile(r"종합 점수:\s*([\d\.]+)점")

        for line in lines:
            score_match = score_pattern.search(line)
            if score_match: result_data["accuracy"] = float(score_match.group(1))

            match = line_pattern.search(line)
            if match:
                time_sec = float(match.group(1))
                msg = match.group(2)
                
                # 📌 [GUI 필터링] 여기서도 0.5초 이하 오차는 굳이 안 띄우도록 거를 수 있음
                # 하지만 일단 모든 데이터를 넘기고, 표시할 때(FeedbackMarker) 필터링하는 게 안전함
                feedback_list.append({"time": time_sec, "msg": msg})
        
        result_data["feedback_list"] = feedback_list

class FeedbackMarker:
    def __init__(self, parent_widget, x_pos, timeline_center_y, message, time_val, all_markers_list, program_label, width=6, max_limit_x=900):
        self.parent = parent_widget
        self.is_open = False
        self.all_markers = all_markers_list
        self.program_label = program_label
        
        self.message = message 
        self.time_val = time_val
        self.base_y = timeline_center_y 
        self.start_x = x_pos
        self.max_limit_x = max_limit_x

        # 막대 생성 (안전장치 포함)
        final_width = width
        if x_pos + final_width > self.max_limit_x:
            final_width = self.max_limit_x - x_pos
            if final_width < 2: final_width = 2

        self.bar = QWidget(parent_widget)
        self.bar.setGeometry(x_pos, timeline_center_y - 20, final_width, 41)
        self.bar.setStyleSheet("background-color: rgb(116, 0, 27); border-radius: 3px;")
        self.bar.show()

        # 버튼 생성
        self.btn = QPushButton("▼", parent_widget)
        btn_x = x_pos + (final_width // 2) - 15
        if btn_x + 30 > self.max_limit_x: btn_x = self.max_limit_x - 30
            
        self.btn.setGeometry(btn_x, timeline_center_y - 50, 30, 30)
        self.btn.setStyleSheet("""
            QPushButton {
                color: rgb(116, 0, 27);
                font: 20px "jura";
                background-color: transparent;
                border: none;
            }
        """)
        self.btn.show()

        self.create_msg_box()
        self.btn.clicked.connect(self.toggle_feedback)

    def create_msg_box(self):
        time_str = seconds_to_min_sec(self.time_val)
        html_content = self.parse_to_html(self.message, time_str)
        
        self.msg_box = QLabel(html_content, self.parent)
        self.msg_box.setGeometry(50, 140, 851, 181) 
        self.msg_box.setStyleSheet("""
            QLabel {
                font: 30px "Pixelify Sans SemiBold";
                color: rgb(116, 0, 27);
                border: 3px solid #74001B;
                border-radius: 15px;
                background-color: transparent;
                padding: 15px;
            }
        """)
        self.msg_box.setWordWrap(True)
        self.msg_box.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.msg_box.hide()

    def update_width(self, new_end_x):
        if new_end_x > self.max_limit_x: new_end_x = self.max_limit_x
        new_width = new_end_x - self.start_x
        if new_width < 6: new_width = 6
        
        current_geo = self.bar.geometry()
        self.bar.setGeometry(self.start_x, current_geo.y(), new_width, current_geo.height())
        
        btn_x = self.start_x + (new_width // 2) - 15
        if btn_x + 30 > self.max_limit_x: btn_x = self.max_limit_x - 30
        self.btn.setGeometry(btn_x, self.base_y - 50, 30, 30)

    def parse_to_html(self, raw_msg, time_str):
        # 📌 [수정] 피드백 표시 조건 강화
        pitch_val = "Correct"
        rhythm_val = "-"
        
        clean_msg = re.sub(r"\s*\(x\d+\)", "", raw_msg)

        # 1. Pitch (틀렸을 때만 표시)
        if "WRONG NOTE" in clean_msg:
            match = re.search(r"Played:\s*([^,]+)", clean_msg)
            wrong = match.group(1).strip() if match else "?"
            pitch_val = f"Wrong Note (Played: {wrong})"
            
            # 박자 정보도 있으면 추출
            r_match = re.search(r"Rhythm:\s*([^)]+)", clean_msg)
            if r_match: rhythm_val = r_match.group(1).strip()
                
        elif "MISS" in clean_msg or "SILENCE" in clean_msg:
            pitch_val = "Missed / Silence"
            rhythm_val = "-"
            
        elif "PERFECT" in clean_msg or "GOOD" in clean_msg or "BAD" in clean_msg:
            # 2. Rhythm (0.5초 이상 차이날 때만 표시)
            diff_val = 0.0
            type_txt = ""
            
            if "Fast" in clean_msg:
                match = re.search(r"Fast\s+([\d\.]+)s", clean_msg)
                if match:
                    diff_val = float(match.group(1))
                    type_txt = "Fast"
            elif "Slow" in clean_msg:
                match = re.search(r"Slow\s+([\d\.]+)s", clean_msg)
                if match:
                    diff_val = float(match.group(1))
                    type_txt = "Slow"
            
            # [핵심] 0.5초 이하는 무시하고 'Correct' 처리 (또는 표시 안 함)
            if diff_val > 0.5:
                rhythm_val = f"{type_txt} ({diff_val}s)"
            else:
                rhythm_val = "Correct"

        count_match = re.search(r"\(x(\d+)\)", raw_msg)
        count_str = f" (x{count_match.group(1)})" if count_match else ""

        style_common = 'font-family:"Pixelify Sans SemiBold"; font-size:30px; color:rgb(116,0,27);'
        style_header = style_common + 'font-weight:600;' 
        style_body = style_common + 'font-weight:400;'
        style_time = 'font-family:"Pixelify Sans SemiBold"; font-size:32px; font-weight:600; color:rgb(116,0,27);'

        html = f"""
        <html>
        <head/>
        <body>
            <p style='line-height:140%'>
                <span style='{style_time}'>[ {time_str} ]{count_str}</span><br>
                <span style='{style_header}'>Pitch:&nbsp;&nbsp;</span>
                <span style='{style_body}'>{pitch_val}</span><br>
                <span style='{style_header}'>Rhythm:</span>
                <span style='{style_body}'>{rhythm_val}</span>
            </p>
        </body>
        </html>
        """
        return html

    def toggle_feedback(self):
        if self.is_open:
            self.close_me()
        else:
            for m in self.all_markers: m.close_me()
            self.btn.setText("▲") 
            self.msg_box.show()
            self.msg_box.raise_()
            self.is_open = True
            if self.program_label: self.program_label.hide()
            
    def close_me(self):
        self.btn.setText("▼")
        self.msg_box.hide()
        self.is_open = False
        if self.program_label:
            self.program_label.show()
            self.program_label.raise_()
            
    def delete(self):
        self.bar.deleteLater()
        self.btn.deleteLater()
        self.msg_box.deleteLater()

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Music Analyzer System")
        self.resize(960, 540)
        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        try:
            self.page1 = uic.loadUi(resource_path("GUI/start.ui"))
            self.page2 = uic.loadUi(resource_path("GUI/Original.ui"))
            self.page2_2 = uic.loadUi(resource_path("GUI/saved_music_list.ui"))
            self.page3 = uic.loadUi(resource_path("GUI/Loading.ui"))
            self.page4 = uic.loadUi(resource_path("GUI/Analyzing.ui"))
            self.page5 = uic.loadUi(resource_path("GUI/Check.ui"))
            self.page6 = uic.loadUi(resource_path("GUI/Analyzing_Loading.ui"))
            self.page7 = uic.loadUi(resource_path("GUI/feedback.ui")) 
        except Exception as e:
            print(f"[UI Load Error] {e}")
            sys.exit()

        self.stack.addWidget(self.page1)
        self.stack.addWidget(self.page2)
        self.stack.addWidget(self.page2_2)
        self.stack.addWidget(self.page3)
        self.stack.addWidget(self.page4)
        self.stack.addWidget(self.page5)
        self.stack.addWidget(self.page6)
        self.stack.addWidget(self.page7)

        self.midi_path = ""
        self.mp3_path = ""
        self.analysis_result = None
        self.feedback_markers = []
        self.static_time_labels = []
        self.history = []

        self.timer_acc = QTimer()
        self.timer_acc.timeout.connect(self.animate_accuracy)
        self.target_acc = 0
        self.current_acc = 0

        self.load_history()

        self.init_page1()
        self.init_page2()
        self.init_page2_2()
        self.init_page3()
        self.init_page4()
        self.init_page5()
        self.init_page6()
        self.init_page7()

    def load_history(self):
        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, 'rb') as f: self.history = pickle.load(f)
            except: self.history = []

    def save_history(self):
        with open(HISTORY_FILE, 'wb') as f: pickle.dump(self.history, f)

    def add_to_history(self, path):
        if path not in self.history:
            self.history.append(path)
            self.save_history()

    def init_page1(self): self.page1.start.clicked.connect(lambda: self.stack.setCurrentIndex(1))
    def init_page2(self):
        self.page2.Back_to_Main.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        self.page2.Upload_Original_File.clicked.connect(self.upload_midi)
        self.page2.Load_Save_File.clicked.connect(self.go_to_saved_list)
    
    def upload_midi(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'MIDI Select', '', 'MIDI Files (*.mid *.midi)')
        if fname:
            self.midi_path = fname
            self.add_to_history(fname)
            self.start_loading_page3()

    def go_to_saved_list(self):
        self.page2_2.list.clear()
        for path in self.history:
            self.page2_2.list.addItem(os.path.basename(path))
        self.stack.setCurrentIndex(2)

    def init_page2_2(self):
        self.page2_2.btn_back.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.page2_2.btn_start.clicked.connect(self.select_from_list)

    def select_from_list(self):
        current_row = self.page2_2.list.currentRow()
        if current_row >= 0:
            self.midi_path = self.history[current_row]
            self.stack.setCurrentIndex(4)
        else:
            QMessageBox.warning(self, "알림", "파일을 선택해주세요.")

    def init_page3(self):
        self.timer_midi = QTimer()
        self.timer_midi.timeout.connect(self.update_midi_loading)
        self.page3.Back_to_Original_From_loading.clicked.connect(self.stop_midi_loading)
    def start_loading_page3(self):
        self.stack.setCurrentIndex(3)
        self.page3.progressBar_loading.setValue(0)
        self.timer_midi.start(30)
    def update_midi_loading(self):
        val = self.page3.progressBar_loading.value()
        if val >= 100:
            self.timer_midi.stop()
            self.stack.setCurrentIndex(4)
        else: self.page3.progressBar_loading.setValue(val + 1)
    def stop_midi_loading(self):
        self.timer_midi.stop()
        self.stack.setCurrentIndex(1)
    def init_page4(self):
        self.page4.Back_to_Original.clicked.connect(lambda: self.stack.setCurrentIndex(1))
        self.page4.Music_Analyzing.clicked.connect(self.upload_mp3)
    def upload_mp3(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'MP3 Select', '', 'Audio Files (*.mp3 *.wav)')
        if fname:
            self.mp3_path = fname
            self.update_check_page()
            self.stack.setCurrentIndex(5)
    def init_page5(self):
        self.page5.Back_to_Music_Analyzing.clicked.connect(lambda: self.stack.setCurrentIndex(4))
        if hasattr(self.page5, 'Lets_Analyze'): self.page5.Lets_Analyze.clicked.connect(self.start_analysis_page6)
    def update_check_page(self):
        self.page5.Original_mp3.setText(f"Original: {os.path.basename(self.midi_path)}")
        self.page5.recorded_mp3.setText(f"User: {os.path.basename(self.mp3_path)}")

    def init_page6(self):
        self.timer_visual = QTimer()
        self.timer_visual.timeout.connect(self.update_visual_progress)
        self.page6.Back_to_Check.clicked.connect(self.stop_analysis)
    def start_analysis_page6(self):
        if not self.midi_path or not self.mp3_path:
            QMessageBox.warning(self, "오류", "파일이 선택되지 않았습니다.")
            return
        self.stack.setCurrentIndex(6)
        self.page6.progressBar_Analyzing.setValue(0)
        self.timer_visual.start(100)
        self.worker = AnalysisWorker(self.midi_path, self.mp3_path)
        self.worker.finished_signal.connect(self.on_analysis_finished)
        self.worker.start()
    def update_visual_progress(self):
        val = self.page6.progressBar_Analyzing.value()
        if val < 95: self.page6.progressBar_Analyzing.setValue(val + 1)
    def stop_analysis(self):
        self.timer_visual.stop()
        if hasattr(self, 'worker'): self.worker.terminate()
        self.stack.setCurrentIndex(5)
    def on_analysis_finished(self, result_data):
        self.timer_visual.stop()
        self.page6.progressBar_Analyzing.setValue(100)
        self.analysis_result = result_data
        QTimer.singleShot(500, self.show_feedback_page)

    def init_page7(self):
        self.page7.homebutton.clicked.connect(self.go_home)
        
        widgets_to_hide = ["triangle1", "triangle2", "feedbackwindow", 
                           "label_4", "label_6", "label_7", "label_9", 
                           "widget_2", "widget_3"]
        for name in widgets_to_hide:
            if hasattr(self.page7, name): getattr(self.page7, name).hide()
            elif self.page7.findChild(QWidget, name): self.page7.findChild(QWidget, name).hide()

        timeline_bar = self.page7.findChild(QWidget, "widget")
        if timeline_bar:
            for child in timeline_bar.findChildren(QLabel): 
                if child.objectName() != "programname": child.hide()
            for child in timeline_bar.findChildren(QPushButton): child.hide()
            for child in timeline_bar.findChildren(QWidget): 
                if child != timeline_bar: child.hide()

    def animate_accuracy(self):
        if self.current_acc < self.target_acc:
            self.current_acc += 1
            self.page7.accuracy.setValue(self.current_acc)
        else:
            self.timer_acc.stop()

    def show_feedback_page(self):
        self.stack.setCurrentIndex(7)
        for marker in self.feedback_markers: marker.delete()
        self.feedback_markers = []
        for lbl in self.static_time_labels: lbl.deleteLater()
        self.static_time_labels = []

        program_lbl = self.page7.findChild(QLabel, "programname")
        if program_lbl:
            program_lbl.show()
            program_lbl.raise_()

        if not self.analysis_result: return

        final_acc = int(self.analysis_result.get("accuracy", 0))
        self.target_acc = final_acc
        self.current_acc = 0
        self.page7.accuracy.setValue(0) 
        self.timer_acc.start(15) 

        timeline_x = 30
        timeline_center_y = 350 + (41 // 2)
        timeline_w = 891
        max_limit_x = timeline_x + timeline_w
        
        # 📌 [수정] 타임라인 그리기 시작점 및 최대 너비 설정 (양쪽 여백 10px)
        draw_start_x = timeline_x + 10 
        draw_width = timeline_w - 20
        max_limit_x = draw_start_x + draw_width
        
        duration = self.analysis_result.get("duration", 1)
        if duration <= 0: duration = 1

        lbl_s = QLabel("00:00", self.page7)
        lbl_s.setGeometry(timeline_x, timeline_center_y + 25, 70, 30)
        lbl_s.setStyleSheet('font: 18px "Pixelify Sans SemiBold"; color: rgb(116, 0, 27);')
        lbl_s.show()
        self.static_time_labels.append(lbl_s)

        lbl_e = QLabel(seconds_to_min_sec(duration), self.page7)
        lbl_e.setGeometry(timeline_x + timeline_w - 70, timeline_center_y + 25, 70, 30)
        lbl_e.setAlignment(Qt.AlignRight)
        lbl_e.setStyleSheet('font: 18px "Pixelify Sans SemiBold"; color: rgb(116, 0, 27);')
        lbl_e.show()
        self.static_time_labels.append(lbl_e)

        feedback_list = self.analysis_result.get("feedback_list", [])
        
        last_x = -999
        last_marker = None
        MIN_DIST = 55 

        for item in feedback_list:
            # 📌 [수정] draw_start_x 기준으로 좌표 계산
            x_pos = int(draw_start_x + ((item["time"] / duration) * draw_width))
            
            # 클램핑
            if x_pos < draw_start_x: x_pos = draw_start_x
            if x_pos > max_limit_x: x_pos = max_limit_x - 5

            if last_marker and (x_pos - last_x < MIN_DIST):
                new_end_x = x_pos + 6
                if (new_end_x - last_marker.start_x) < 150:
                    last_marker.update_width(new_end_x)
                else:
                    marker = FeedbackMarker(self.page7, x_pos, timeline_center_y, 
                                            item["msg"], item["time"], self.feedback_markers, program_lbl, width=6, max_limit_x=max_limit_x)
                    self.feedback_markers.append(marker)
                    last_marker = marker
                    last_x = x_pos
            else:
                marker = FeedbackMarker(self.page7, x_pos, timeline_center_y, 
                                        item["msg"], item["time"], self.feedback_markers, program_lbl, width=6, max_limit_x=max_limit_x)
                self.feedback_markers.append(marker)
                last_marker = marker
                last_x = x_pos

    def go_home(self):
        self.midi_path = ""
        self.mp3_path = ""
        self.analysis_result = None
        self.timer_acc.stop()
        self.stack.setCurrentIndex(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindow = MainApp()
    myWindow.show()
    sys.exit(app.exec_())