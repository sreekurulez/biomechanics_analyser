import cv2

# Video capture & writing functions
def open_video(input_path):
    """Open a video file and return capture object, width, height, and fps."""
    cap = cv2.VideoCapture(input_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return cap, frame_width, frame_height, fps

def create_writer(output_path, fourcc, fps, frame_size):
    """Create a video writer object for output video."""
    return cv2.VideoWriter(output_path, fourcc, fps, frame_size)
