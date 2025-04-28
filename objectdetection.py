import streamlit as st
from streamlit_option_menu import option_menu
import cv2
from ultralytics import YOLO
import torch
import numpy as np
import tempfile
import os
import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.palettes import Category20, Category10
import json
import boto3
from datetime import datetime

# Streamlit page config
st.set_page_config(page_title="Object Detection in Video", layout="wide")

# Sidebar Option Menu
with st.sidebar:
    selected = option_menu(
        menu_title="Menu",
        options=["About", "Video Detection", "Live Detection", "Objects Detected", "Graphs"],
        icons=["info-circle", "camera-video", "camera-reels", "clipboard-data", "bar-chart"],
        menu_icon="cast",
        default_index=0,
        orientation="vertical",
        styles={
            "container": {"padding": "5px"},
            "icon": {"color": "white", "font-size": "20px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "crimson",
            },
            "nav-link-selected": {"background-color": "#02ab21", "color": "white"},
        },
    )

# Load labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Dangerous objects for alert
dangerous_objects = ["knife", "gun", "bomb"]

# SNS Alert function
def send_sns_alert(object_name):
    aws_access_key = st.secrets["aws"]["aws_access_key"]
    aws_secret_key = st.secrets["aws"]["aws_secret_key"]
    region_name = st.secrets["aws"]["region_name"]
    topic_arn = st.secrets["aws"]["sns_topic_arn"]

    client = boto3.client("sns",
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
        region_name=region_name
    )

    message = f"⚠️ ALERT: Dangerous object '{object_name}' detected in the video stream!"
    subject = "🔴 Object Detection Alert"

    try:
        client.publish(TopicArn=topic_arn, Message=message, Subject=subject)
        st.warning(message)
    except Exception as e:
        st.error(f"❌ Failed to send SNS alert: {e}")

# Upload to S3
@st.cache_data
def upload_to_s3(detected_objects_dict):
    aws_access_key = st.secrets["aws"]["aws_access_key"]
    aws_secret_key = st.secrets["aws"]["aws_secret_key"]
    bucket_name = st.secrets["aws"]["bucket_name"]
    region_name = st.secrets["aws"]["region_name"]

    output_data_path = os.path.join(tempfile.gettempdir(), "detected_objects.json")
    with open(output_data_path, 'w') as f:
        json.dump(detected_objects_dict, f)

    s3 = boto3.client("s3",
                      aws_access_key_id=aws_access_key,
                      aws_secret_access_key=aws_secret_key,
                      region_name=region_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    s3_file_name = f"detected_objects_{timestamp}.json"

    try:
        with open(output_data_path, "rb") as f:
            s3.upload_fileobj(f, bucket_name, s3_file_name)
        st.success(f"✅ Detected objects data uploaded to S3 as: `{s3_file_name}`")
    except Exception as e:
        st.error(f"❌ Failed to upload detected objects data to S3: {e}")



if selected == "About":
    st.markdown("""
        <style>
            body {
                background-color: black;
                color: white;
            }
            .animated-heading {
                font-size: 32px;
                font-weight: bold;
                color: #02ab21;
                text-align: center;
                margin-top: 20px;
                margin-bottom: 30px;
                animation: fadeSlideIn 2s ease-in-out;
            }

            @keyframes fadeSlideIn {
                0% { opacity: 0; transform: translateY(-30px); }
                100% { opacity: 1; transform: translateY(0); }
            }

            .about-container {
                background-color: #f4f9f9;
                padding: 30px;
                border-radius: 15px;
                box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                text-align: center;
                animation: fadeIn 1.5s ease-in-out;
                margin-top: 20px;
                margin-bottom: 20px;
            }

            @keyframes fadeIn {
                from { opacity: 0; transform: scale(0.95); }
                to { opacity: 1; transform: scale(1); }
            }

            .about-container p {
                font-size: 18px;
                color: #333;
                line-height: 1.6;
                margin-top: 20px;
                margin-bottom: 30px; /* Ensure space below text */
            }

            .tech-button {
                display: inline-block;
                background-color: crimson;
                color: white;
                padding: 4px 12px;
                border-radius: 15px;
                font-size: 15px;
                font-weight: 500;
                margin: 8px; /* Spacing between buttons */
                box-shadow: 0 3px 6px rgba(0,0,0,0.1);
            }

            .tech-buttons-container {
                display: flex;
                justify-content: center;
                flex-wrap: wrap;
                gap: 10px;  /* Spacing between buttons */
                margin-top: 20px;
            }
        </style>

        <div class="animated-heading">Object Detection in Real-Time Video</div>

        <div class="about-container">
            <p>
                Welcome to the Object Detection in Real-Time Video using<br>
            </p>
            <div class="tech-buttons-container">
                <span class="tech-button">YOLOv5</span>
                <span class="tech-button">YOLOv8</span>
                <span class="tech-button">OpenCV</span><br>
            </div>
            <p> This application allows you to upload videos, analyze them live with object detection, 
                and view powerful visual insights from the detections and store them in AWS-S3. </p>
        </div>
    """, unsafe_allow_html=True)
















# Video Detection Section
elif selected == "Video Detection":
    st.markdown("<h1 style='color:#02ab21;'>Objects Detection In Video</h1>", unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)

        model = YOLO("yolov8n.pt")
        detected_objects_dict = {}
        stframe = st.empty()
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_bar = st.progress(0)

        frame_id = 0
        if 'alert_sent' not in st.session_state:
            st.session_state['alert_sent'] = False

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_id += 1
            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    detected_objects_dict.setdefault(label, []).append(conf * 100)

                    # Trigger alert
                    if label.lower() in dangerous_objects and not st.session_state['alert_sent']:
                        send_sns_alert(label)
                        st.session_state['alert_sent'] = True

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            progress_bar.progress(min(frame_id / total_frames, 1.0))

        cap.release()
        os.remove(tfile.name)
        st.session_state["detected_data"] = detected_objects_dict
        upload_to_s3(detected_objects_dict)

# Live Detection Section
elif selected == "Live Detection":
    st.markdown("<h1 style='color:#02ab21;'>Real-Time Object Detection (YOLOv5)</h1>", unsafe_allow_html=True)
    model = YOLO("yolov5s.pt")
    start_btn = st.button("Start Webcam Detection")

    if start_btn:
        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        detected_objects_dict = {}
        if 'alert_sent' not in st.session_state:
            st.session_state['alert_sent'] = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = model(frame, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    detected_objects_dict.setdefault(label, []).append(conf * 100)

                    # Trigger alert
                    if label.lower() in dangerous_objects and not st.session_state['alert_sent']:
                        send_sns_alert(label)
                        st.session_state['alert_sent'] = True

            stframe.image(frame, channels="BGR", use_column_width=True)

        cap.release()
        st.session_state["detected_data"] = detected_objects_dict
        upload_to_s3(detected_objects_dict)

# Objects Detected Section
elif selected == "Objects Detected":
    st.markdown("<h1 style='color:#02ab21;'>Detected Objects</h1>", unsafe_allow_html=True)
    detected_objects_dict = st.session_state.get("detected_data", {})

    if detected_objects_dict:
        sorted_objects = sorted(detected_objects_dict.items(), key=lambda x: -len(x[1]))
        for obj, acc_list in sorted_objects:
            avg_acc = round(sum(acc_list) / len(acc_list), 2)
            st.write(f"🔸 **{obj}** — Average Accuracy: {avg_acc}% (Detected {len(acc_list)} times)")
    else:
        st.info("Run the Video Detection section first to see detected objects.")

# Graphs Section
elif selected == "Graphs":
    st.markdown("<h1 style='color:#02ab21;'>Visualization of Objects</h1>", unsafe_allow_html=True)
    detected_objects_dict = st.session_state.get("detected_data", {})

    if detected_objects_dict:
        data = {
            "Object": list(detected_objects_dict.keys()),
            "Frequency": [len(acc_list) for acc_list in detected_objects_dict.values()],
            "Avg Accuracy": [round(sum(acc_list) / len(acc_list), 2) for acc_list in detected_objects_dict.values()]
        }
        df = pd.DataFrame(data)

        # Plotly
        import plotly.express as px
        st.subheader("📉 Plotly: Interactive Bar Chart")
        fig3 = px.bar(df, x='Object', y='Frequency', color='Avg Accuracy',
                      title="Plotly - Object Frequency and Accuracy", text='Frequency')
        fig3.update_layout(xaxis_title="Object", yaxis_title="Frequency", title_x=0.5)
        st.plotly_chart(fig3)

        # Altair
        import altair as alt
        st.subheader("📊 Altair: Accuracy Line Chart")
        chart = alt.Chart(df).mark_line(point=True).encode(
            x='Object',
            y='Avg Accuracy',
            tooltip=['Object', 'Avg Accuracy']
        ).properties(title="Altair - Average Accuracy Trend")
        st.altair_chart(chart, use_container_width=True)

        # Matplotlib
        st.subheader("📊 Matplotlib: Pie Chart")
        fig1, ax1 = plt.subplots()
        ax1.pie(df["Frequency"], labels=df["Object"], autopct='%1.1f%%', startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

        # Seaborn
        import seaborn as sns
        st.subheader("📈 Seaborn: Barplot - Object Frequency")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x="Frequency", y="Object", data=df, ax=ax2, palette="viridis")
        ax2.set_title("Object Frequency by Detection Count")
        st.pyplot(fig2)

        st.subheader("🔢 Seaborn: Count Plot - Object Frequency")
        expanded_data = []
        for obj, confs in detected_objects_dict.items():
            for conf in confs:
                expanded_data.append({"Object": obj, "Confidence": conf})
        expanded_df = pd.DataFrame(expanded_data)

        fig_count, ax_count = plt.subplots(figsize=(10, 6))
        sns.countplot(data=expanded_df, y="Object", order=df.sort_values("Frequency", ascending=False)["Object"])
        ax_count.set_title("Detected Object Frequency (Raw Count)")
        st.pyplot(fig_count)

        st.subheader("📦 Seaborn: Box Plot - Confidence Score Distribution")
        fig_box, ax_box = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=expanded_df, x="Object", y="Confidence", palette="Set2")
        ax_box.set_title("Confidence Score Distribution by Object")
        ax_box.set_xticklabels(ax_box.get_xticklabels(), rotation=45)
        st.pyplot(fig_box)

    else:
        st.warning("No detection data available. Please upload and process a video first.")

