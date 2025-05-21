# Smart-Surveillance-System-with-Real-Time-Threat-Detection


## 📌 Project Overview
#### The Smart Surveillance System with Real Time Threat Detection is an AI-powered security solution designed to enhance monitoring and safety through intelligent, automated video analysis. Built using advanced deep learning models like YOLOv5 and YOLOv8, the system performs real-time object detection on both live webcam feeds and uploaded video footage. It identifies and tracks objects with high precision, and is capable of detecting potentially dangerous items such as knives, guns, or bombs. Upon detecting such threats, the system immediately triggers real-time alerts via AWS SNS (Simple Notification Service), ensuring a quick response. Detection results and logs are automatically uploaded to AWS S3 for secure cloud storage and future analysis. The system also offers a user-friendly interface built with Streamlit, featuring clear navigation and interactive visualizations created with libraries like Seaborn, Plotly, Matplotlib, and Altair. These visual tools help users easily interpret object counts, detection frequency, and patterns. Overall, the Smart Surveillance System combines AI, cloud services, and data visualization to provide a robust and scalable solution for modern security challenges.
---

## ✨ Key Features
- **Real-Time Object Detection**: Detects multiple objects (people, vehicles, weapons, etc.) in live webcam feeds.
- **Dangerous Object Alerts**: Sends **instant alerts** using **AWS SNS** when dangerous objects like knives, guns, or bombs are detected.
- **AWS S3 Integration**: Saves detection results (object data) automatically to **AWS S3** for secure storage and backup.
- **Data Visualization and Analytics**: Provides interactive graphs, charts, and summaries for analysis of detected objects using **Plotly**, **Seaborn**, and **Matplotlib**.


---

## 📂 Technologies Used
- **Python**
- **YOLOv5**, **YOLOv8** (Ultralytics for object detection)
- **OpenCV** (Real-time video processing)
- **Streamlit** (Web application framework)
- **AWS S3** (Data storage)
- **AWS SNS** (Sending alerts)
- **Plotly**, **Seaborn**, **Matplotlib** (Data visualization)

---
