import cv2
import pytesseract
import numpy as np
from pathlib import Path
import re

# I edit videos on Windows, this is why I have that here. Change depends on your needs
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class FPSAnalyzer:
    def __init__(self, video_path):
        self.video_path = Path(video_path)
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.cap = cv2.VideoCapture(str(video_path))
        self.fps_values = []

    def preprocess_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        return thresh

    def extract_fps_from_text(self, text):
        numbers = re.findall(r'\d+\.?\d*', text)
        fps_candidates = [float(num) for num in numbers if float(num) < 1000]
        return fps_candidates[0] if fps_candidates else None

    def analyze(self, progress_callback=None):
        if not self.cap.isOpened():
            raise Exception("Error opening video file")

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        processed_frames = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            processed_frames += 1
            if progress_callback:
                progress = (processed_frames / total_frames) * 100
                progress_callback(progress)

            processed = self.preprocess_frame(frame)

            text = pytesseract.image_to_string(processed, config='--psm 6')

            fps = self.extract_fps_from_text(text)
            if fps is not None:
                self.fps_values.append(fps)

        self.cap.release()

    def get_results(self):
        if not self.fps_values:
            return {
                'average_fps': None,
                'min_fps': None,
                'max_fps': None,
                'samples_count': 0
            }

        return {
            'average_fps': np.mean(self.fps_values),
            'min_fps': min(self.fps_values),
            'max_fps': max(self.fps_values),
            'samples_count': len(self.fps_values)
        }


def print_progress(progress):
    print(f"\rProgress: {progress:.1f}%", end='')


def main():
    video_path = input("Enter the path to your video file: ")

    try:
        analyzer = FPSAnalyzer(video_path)

        print("\nAnalyzing video...")
        analyzer.analyze(progress_callback=print_progress)

        results = analyzer.get_results()

        print("\n\nResults:")
        print(f"Average FPS: {results['average_fps']:.2f}")
        print(f"Minimum FPS: {results['min_fps']:.2f}")
        print(f"Maximum FPS: {results['max_fps']:.2f}")
        print(f"Number of samples: {results['samples_count']}")

    except Exception as e:
        print(f"\nError: {str(e)}")


if __name__ == "__main__":
    main()
