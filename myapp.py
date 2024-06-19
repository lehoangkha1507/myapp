import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
from exercise import camera

model = YOLO("best.pt")

# Set the default time for work and break intervals
WORK_TIME = 25 * 60
SHORT_BREAK_TIME = 5 * 60
LONG_BREAK_TIME = 15 * 60

# Function to convert seconds to minutes and seconds
def format_time(seconds):
    minutes, seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{seconds:02d}"

# Initialize state variables
if "work_time" not in st.session_state:
    st.session_state.work_time = WORK_TIME
if "break_time" not in st.session_state:
    st.session_state.break_time = SHORT_BREAK_TIME
if "is_work_time" not in st.session_state:
    st.session_state.is_work_time = True
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "pomodoros_completed" not in st.session_state:
    st.session_state.pomodoros_completed = 0

def start_timer():
    st.session_state.is_running = True

def stop_timer():
    st.session_state.is_running = False

st.title("Pomodoro Timer")

# Timer display
timer_placeholder = st.empty()
timer_placeholder.text(format_time(st.session_state.work_time if st.session_state.is_work_time else st.session_state.break_time))

# Start/Stop buttons
start_button = st.button("Start", on_click=start_timer, disabled=st.session_state.is_running)
stop_button = st.button("Stop", on_click=stop_timer, disabled=not st.session_state.is_running)

# Video feed
frame_placeholder = st.empty()

def update_timer():
    if st.session_state.is_running:
        if st.session_state.is_work_time:
            st.session_state.work_time -= 1
            if st.session_state.work_time == 0:
                st.session_state.is_work_time = False
                st.session_state.pomodoros_completed += 1
                st.session_state.break_time = LONG_BREAK_TIME if st.session_state.pomodoros_completed % 4 == 0 else SHORT_BREAK_TIME
                st.success("Take a long break and rest your mind." if st.session_state.pomodoros_completed % 4 == 0 else "Take a short break and stretch your legs!")
        else:
            st.session_state.break_time -= 1
            if st.session_state.break_time == 0:
                st.session_state.is_work_time = True
                st.session_state.work_time = WORK_TIME
                st.info("Get back to work!")

        timer_placeholder.text(format_time(st.session_state.work_time if st.session_state.is_work_time else st.session_state.break_time))
        st.experimental_rerun()

def run_model():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video.")
        stop_timer()
        return

    results = model(frame)

    for result in results:
        boxes = result.boxes
        for bbox in boxes.xyxy:
            xmin, ymin, xmax, ymax = bbox
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
        if len(boxes.xyxy) > 0:
            if result.names[int(boxes.cls.cpu().numpy()[0])] == 'drowsy':
                stop_timer()
                st.error("You are drowsy. Take a break!")
                answer = st.button("Would you like to exercise a little to be more awake?")
                if answer:
                    camera()
                    st.success("Let's continue studying")
                break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame)
    frame_placeholder.image(img)

if st.session_state.is_running:
    update_timer()
    run_model()
