import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
import io
import base64

# Cache the model to prevent reloading
@st.cache_resource
def load_emotion_model():
    # Load the entire model (architecture + weights)
    emotion_model = load_model('emotion_model4_v23.h5')
    return emotion_model


# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

def detect_emotion(frame, emotion_model):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
    
    for (x, y, w, h) in num_faces:
        roi_gray_frame = gray_frame[y:y+h, x:x+w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        return emotion_dict[maxindex], (x, y, w, h), emotion_prediction
    return None, None, None

# Radar chart creation function
def create_radar_chart(emotion_prediction):
    labels = list(emotion_dict.values())
    values = emotion_prediction.flatten().tolist()

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]

    # Resize radar chart to fit within the same line as live camera feed
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))  # Adjusted to fit with camera feed size
    ax.fill(angles, values, color='red', alpha=0.25)
    ax.plot(angles, values, color='red', linewidth=2)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Save the plot as an image and convert it to base64
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    radar_chart_img_str = base64.b64encode(buf.read()).decode('utf-8')
    return radar_chart_img_str

def main():
    st.title("Facial Emotion Recognition")

    # Load emotion model
    emotion_model = load_emotion_model()

    # User choice: Upload Image or Live Camera
    choice = st.radio("Choose input method", ("Upload Photo", "Use Live Camera"))

    if choice == "Upload Photo":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            frame = np.array(image)
            emotion, face_location, emotion_prediction = detect_emotion(frame, emotion_model)
            if emotion:
                x, y, w, h = face_location
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 4)
                
                # Get image height and width for positioning the text at the bottom
            height, width, _ = frame.shape
            font_scale = 0.8  # Smaller font size
            white_text_thickness = 1  # Lighter white text
            black_stroke_thickness = 4  # Wider black stroke
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Position the text at the bottom center of the image
            text = f"Detected Emotion: {emotion}"
            text_size, _ = cv2.getTextSize(text, font, font_scale, white_text_thickness)
            text_x = (width - text_size[0]) // 2  # Center horizontally
            text_y = height - 20  # Position at bottom with some padding

            # Draw black stroke (outline)
            frame = cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                                (0, 0, 0), black_stroke_thickness, cv2.LINE_AA)

            # Draw white text on top of the black stroke
            frame = cv2.putText(frame, text, (text_x, text_y), font, font_scale, 
                                (255, 255, 255), white_text_thickness, cv2.LINE_AA)

            st.image(frame, caption=f"Detected Emotion: {emotion}", use_column_width=True)

            # Display radar chart
            radar_chart_img_str = create_radar_chart(emotion_prediction)
            st.markdown(f'<img src="data:image/png;base64,{radar_chart_img_str}"/>', unsafe_allow_html=True)
            
        else:
            st.write("No face detected.")
    
    elif choice == "Use Live Camera":
        run = st.checkbox('Run')
        col1, col2 = st.columns([1, 1])  # Two equal columns to fit camera feed and radar chart side by side
        FRAME_WINDOW = col1.image([])
        radar_chart_container = col2.empty()  # Container for radar chart
        camera = cv2.VideoCapture(0)

        while run:
            _, frame = camera.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect emotion
            emotion, face_location, emotion_prediction = detect_emotion(frame, emotion_model)
            if emotion:
                x, y, w, h = face_location
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 4)
                frame = cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Show the radar chart live in the right column
                radar_chart_img_str = create_radar_chart(emotion_prediction)
                radar_chart_container.markdown(f'<img src="data:image/png;base64,{radar_chart_img_str}"/>',
                                               unsafe_allow_html=True)

            # Display the frame in the left column
            FRAME_WINDOW.image(frame, width=450)  # Keep fixed size for the webcam feed

        camera.release()

if __name__ == "__main__":
    main()