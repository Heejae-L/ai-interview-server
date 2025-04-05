import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread
from rich.console import Console
import os
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from time import sleep
from PIL import Image, ImageTk  # Import for image handling
import importlib.util  # 동적 모듈 로드를 위한 모듈

# 인터뷰 관련 모듈 가져오기
from interviewFIN2 import (
    ResumeJsonParser,
    InterviewQuestionMaker,
    record_audio,
    transcribe_audio,
    VideoRecorder,
)

console = Console()

def run_pose_detection(root):
    """
    1212_correct_pose_detection_video.py 모듈을 동적으로 로드하여 
    인터뷰 영상("interview_recording.avi" 혹은 생성된 영상 파일)에 대해 자세 분석 기능을 수행합니다.
    영상 분석이 완료되면 메인 스레드를 통해 알림창을 띄우고 프로그램을 종료합니다.
    """
    spec = importlib.util.spec_from_file_location("pose_detection", "1212_correct_pose_detection_video.py")
    pose_detection = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pose_detection)
    # 인터뷰 영상 파일이 생성되었다고 가정하고, 해당 영상에 대해 pose_detection.main()을 실행합니다.
    pose_detection.main()
    # 영상 분석이 완료되면 메인 스레드에서 알림창을 표시한 후 프로그램 종료
    root.after(0, lambda: (
        messagebox.showinfo("Analysis Completed", "Pose detection analysis is complete. The program will now exit."),
        root.quit()
    ))

class InterviewApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Mock Interview Application")
        self.root.state('zoomed')  # Set window to fullscreen

        # Centering the UI
        self.main_frame = tk.Frame(root)
        self.main_frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        # UI Elements
        self.resume_path_label = tk.Label(self.main_frame, text="Select Resume (PDF):")
        self.resume_path_label.grid(row=0, column=0, padx=10, pady=10)

        self.resume_path_entry = tk.Entry(self.main_frame, width=50)
        self.resume_path_entry.grid(row=0, column=1, padx=10, pady=10)

        self.browse_button = tk.Button(self.main_frame, text="Browse", command=self.browse_file)
        self.browse_button.grid(row=0, column=2, padx=10, pady=10)

        self.output_video_label = tk.Label(self.main_frame, text="Output Video File:")
        self.output_video_label.grid(row=1, column=0, padx=10, pady=10)

        self.output_video_entry = tk.Entry(self.main_frame, width=50)
        self.output_video_entry.insert(0, "interview_recording.avi")
        self.output_video_entry.grid(row=1, column=1, padx=10, pady=10)

        self.start_button = tk.Button(self.main_frame, text="Start Interview", command=self.start_interview_thread)
        self.start_button.grid(row=2, column=1, padx=10, pady=10)

        self.progress_label = tk.Label(self.main_frame, text="Status: Waiting to start...")
        self.progress_label.grid(row=3, column=0, columnspan=3, padx=10, pady=10)

        self.question_label = tk.Label(self.main_frame, text="", font=("Arial", 14), wraplength=600, justify="center")
        self.question_label.grid(row=4, column=0, columnspan=3, padx=10, pady=20)

        self.avatar_label = tk.Label(self.main_frame, text="Avatar Interviewer:")
        self.avatar_label.grid(row=5, column=0, columnspan=3)
        self.avatar_panel = tk.Label(self.main_frame)
        self.avatar_panel.grid(row=6, column=0, columnspan=3)

        # Load Avatar Images
        self.normal_avatar_path = "stop.png"
        self.speaking_avatar_path = "start.png"
        self.normal_avatar = ImageTk.PhotoImage(Image.open(self.normal_avatar_path).resize((800, 600)))
        self.speaking_avatar = ImageTk.PhotoImage(Image.open(self.speaking_avatar_path).resize((800, 600)))

        # Display the default avatar
        self.avatar_panel.config(image=self.normal_avatar)

        self.video_recorder = None
        self.stop_interview = False

    def browse_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
        if file_path:
            self.resume_path_entry.delete(0, tk.END)
            self.resume_path_entry.insert(0, file_path)

    def start_interview_thread(self):
        self.stop_interview = False
        Thread(target=self.start_interview, daemon=True).start()

    def start_interview(self):
        resume_path = self.resume_path_entry.get()
        output_video = self.output_video_entry.get()

        if not resume_path or not os.path.isfile(resume_path):
            messagebox.showerror("Error", "Please select a valid resume file.")
            return

        self.progress_label.config(text="Status: Starting the interview process...")
        json_parser = ResumeJsonParser()
        question_maker = InterviewQuestionMaker()
        self.video_recorder = VideoRecorder(output_file=output_video)
        self.video_recorder.start_recording()

        try:
            self.progress_label.config(text="Status: Parsing resume...")
            resume_json = json_parser.pdf2json(resume_path)
            questions = question_maker.create_questions(resume_path)

            if isinstance(questions, dict):
                all_questions = questions.get("technical_questions", []) + questions.get("behavior_questions", [])
            else:
                messagebox.showerror("Error", "Failed to generate questions.")
                return

            for idx, question in enumerate(all_questions, 1):
                if self.stop_interview:
                    break
                self.ask_question(question, idx)
        finally:
            self.video_recorder.stop_recording()
            self.progress_label.config(text=f"Status: Recording stopped and saved as {output_video}.")
            # 인터뷰 종료 후, 자세 분석 단계 실행
            self.finish_interview()

    def ask_question(self, question, idx):
        # Update avatar to speaking state
        self.avatar_panel.config(image=self.speaking_avatar)
        self.question_label.config(text=f"Question {idx}: {question}")
        tts = gTTS(text=question, lang='ko')
        mp3_filename = f"question_{idx}.mp3"
        tts.save(mp3_filename)
        audio = AudioSegment.from_mp3(mp3_filename)
        play(audio)

        # Record response
        self.progress_label.config(text="Status: Recording your response...")
        response_path = record_audio()
        self.progress_label.config(text="Status: Transcribing response...")
        response_text = transcribe_audio(response_path)
        console.print(f"[bold yellow]User Response: {response_text}[/bold yellow]")

        # Check for user exit
        if "그만하겠습니다" in response_text:
            self.stop_interview = True
            messagebox.showinfo("Info", "Interview stopped by user request.")

        # Clean up and restore avatar
        os.remove(mp3_filename)
        os.remove(response_path)
        self.avatar_panel.config(image=self.normal_avatar)

    def finish_interview(self):
        """
        인터뷰 종료 후, 알림창이 표시된 후 사용자가 확인하면
        1212_correct_pose_detection_video.py의 자세 분석 기능을 실행하는 창이 생성되고,
        분석 종료 후 "분석 완료" 알림창과 함께 프로그램이 종료됩니다.
        """
        # 인터뷰 종료 알림창 표시
        messagebox.showinfo("Interview Completed", "Interview finished. Starting pose detection on the generated video. Click OK to proceed.")
        console.log("Interview finished. Starting pose detection on the generated video.")
        # 새로운 스레드에서 pose detection 실행 (cv2 창이 생성됨)
        Thread(target=lambda: run_pose_detection(self.root), daemon=True).start()

if __name__ == "__main__":
    root = tk.Tk()
    app = InterviewApp(root)
    root.mainloop()