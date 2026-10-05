import sys
from pathlib import Path

import streamlit as st

# ---------------------------------------------------------
# Make project root available
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module_05_ppe.inference.pipeline import process_video


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="NEXORA - PPE Safety",
    page_icon="🛡️",
    layout="wide"
)


# ---------------------------------------------------------
# Light Blue Theme
# ---------------------------------------------------------
st.markdown("""
<style>

.stApp {
    background-color: #F4FAFF;
}

/* Top spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}

/* Headings */
h1 {
    color: #0D47A1 !important;
    font-weight: 800;
}

h2 {
    color: #1565C0 !important;
}

h3 {
    color: #1976D2 !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #BBDEFB;
    border-radius: 14px;
    padding: 18px;
    box-shadow: 0 4px 12px rgba(25, 118, 210, 0.08);
}

[data-testid="stMetricLabel"] {
    color: #546E7A !important;
}

[data-testid="stMetricValue"] {
    color: #1565C0 !important;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background-color: #1976D2;
    color: white;
    border: none;
    border-radius: 9px;
    padding: 10px 24px;
    font-weight: 700;
}

.stButton > button:hover {
    background-color: #0D47A1;
    color: white;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white;
    border: 1px solid #BBDEFB;
    border-radius: 14px;
    padding: 12px;
}

/* Inputs */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #90CAF9;
}

/* Select boxes */
.stSelectbox > div > div {
    border-radius: 8px;
}

/* Alert cards */
.alert-card {
    background: white;
    border-radius: 14px;
    padding: 18px;
    margin: 10px 0;
    border-left: 5px solid #1976D2;
    box-shadow: 0 3px 10px rgba(25, 118, 210, 0.08);
}

.critical {
    border-left-color: #D32F2F;
}

.high {
    border-left-color: #F57C00;
}

.medium {
    border-left-color: #1976D2;
}

/* Header */
.dashboard-header {
    background: linear-gradient(135deg, #E3F2FD, #F4FAFF);
    border: 1px solid #BBDEFB;
    border-radius: 18px;
    padding: 25px 30px;
    margin-bottom: 25px;
}

.dashboard-title {
    font-size: 34px;
    font-weight: 800;
    color: #0D47A1;
}

.dashboard-subtitle {
    color: #546E7A;
    font-size: 16px;
    margin-top: 5px;
}

/* Section */
.section-title {
    color: #1565C0;
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

/* Status */
.status-online {
    color: #2E7D32;
    font-weight: 700;
}

</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">🛡️ NEXORA PPE SAFETY MONITORING</div>
    <div class="dashboard-subtitle">
        AI-powered workplace safety monitoring • Helmet & Safety Vest Detection
    </div>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:

    st.markdown("## ⚙️ Control Panel")

    camera_id = st.text_input(
        "Camera ID",
        value="CAM-002"
    )

    st.markdown("---")

    st.markdown("### 🎥 Video Input")

    uploaded_video = st.file_uploader(
        "Upload workplace video",
        type=["mp4", "mov", "avi", "mkv"]
    )

    st.markdown("---")

    st.markdown("### ⚡ Processing")

    quick_test = st.checkbox(
        "Quick test",
        value=False,
        help="Process only the first 300 frames."
    )

    if quick_test:
        max_frames = 300
    else:
        max_frames = None

    st.markdown("---")

    st.markdown(
        '<div class="status-online">● Detection Engine Ready</div>',
        unsafe_allow_html=True
    )


# ---------------------------------------------------------
# Main upload area
# ---------------------------------------------------------
st.markdown(
    '<div class="section-title">📹 Video Analysis</div>',
    unsafe_allow_html=True
)

if uploaded_video is None:

    st.info(
        "Upload a workplace CCTV/video file from the left panel to start PPE monitoring."
    )

else:

    st.success(
        f"Video loaded: **{uploaded_video.name}**"
    )

    col1, col2 = st.columns([3, 1])

    with col1:
        st.video(uploaded_video)

    with col2:
        st.markdown("### 🎬 Analysis")
        st.write(f"**Camera:** {camera_id}")
        st.write(f"**File:** {uploaded_video.name}")

        if st.button(
            "🚀 Start PPE Detection",
            use_container_width=True
        ):

            # Save uploaded video temporarily
            temp_dir = PROJECT_ROOT / "videos"
            temp_dir.mkdir(exist_ok=True)

            input_path = temp_dir / uploaded_video.name

            with open(input_path, "wb") as f:
                f.write(uploaded_video.getbuffer())

            try:

                with st.spinner(
                    "AI is analyzing the video... Please wait."
                ):

                    result = process_video(
                        source=str(input_path),
                        camera_id=camera_id,
                        save_annotated=True,
                        max_frames=max_frames
                    )

                st.session_state.result = result
                st.session_state.output_dir = Path(
                    result["files"]["output_dir"]
                )

                st.success("✅ Video analysis completed!")

            except Exception as e:

                st.error(
                    f"Detection failed: {e}"
                )


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------
result = st.session_state.result

if result is not None:

    events = result.get("events", [])

    st.markdown(
        '<div class="section-title">📊 Safety Overview</div>',
        unsafe_allow_html=True
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    total_events = len(events)

    critical_events = sum(
        1 for e in events
        if e.get("severity") == "CRITICAL"
    )

    high_events = sum(
        1 for e in events
        if e.get("severity") == "HIGH"
    )

    medium_events = sum(
        1 for e in events
        if e.get("severity") == "MEDIUM"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🚨 Total Events",
            total_events
        )

    with col2:
        st.metric(
            "🔴 Critical",
            critical_events
        )

    with col3:
        st.metric(
            "🟠 High",
            high_events
        )

    with col4:
        st.metric(
            "🔵 Medium",
            medium_events
        )


    # -----------------------------------------------------
    # Processing information
    # -----------------------------------------------------
    st.markdown(
        '<div class="section-title">📈 Processing Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Camera",
            result.get("camera_id", camera_id)
        )

    with col2:
        st.metric(
            "Frames",
            result.get("frames_processed", 0)
        )

    with col3:
        st.metric(
            "FPS",
            result.get("video_fps", 0)
        )

    with col4:
        processing = result.get("processing", {})

        if isinstance(processing, dict):
            processing_time = processing.get(
                "elapsed_seconds",
                processing.get("seconds", 0)
            )
        else:
            processing_time = processing

        st.metric(
            "Processing",
            f"{processing_time}s"
        )


    # -----------------------------------------------------
    # Annotated video
    # -----------------------------------------------------
    output_dir = st.session_state.output_dir

    annotated_video = output_dir / "annotated_video.mp4"

    if annotated_video.exists():

        st.markdown(
            '<div class="section-title">🎥 Detection Result</div>',
            unsafe_allow_html=True
        )

        st.video(str(annotated_video))

        with open(annotated_video, "rb") as video_file:

            st.download_button(
                label="⬇️ Download Annotated Video",
                data=video_file,
                file_name="NEXORA_PPE_Annotated.mp4",
                mime="video/mp4"
            )


    # -----------------------------------------------------
    # Events
    # -----------------------------------------------------
    st.markdown(
        '<div class="section-title">🚨 Detected Safety Events</div>',
        unsafe_allow_html=True
    )

    if len(events) == 0:

        st.success(
            "✅ No PPE violations detected in this video."
        )

    else:

        for event in events:

            severity = event.get(
                "severity",
                "UNKNOWN"
            )

            if severity == "CRITICAL":
                css_class = "critical"
                icon = "🔴"

            elif severity == "HIGH":
                css_class = "high"
                icon = "🟠"

            else:
                css_class = "medium"
                icon = "🔵"

            missing = event.get(
                "missing_ppe",
                []
            )

            missing_text = ", ".join(
                str(x).replace("_", " ").title()
                for x in missing
            )

            st.markdown(
                f"""
                <div class="alert-card {css_class}">
                    <h3>{icon} {severity} — {event.get("event_id", "Event")}</h3>
                    <b>Track:</b> {event.get("track_id", "N/A")}<br>
                    <b>Missing PPE:</b> {missing_text}<br>
                    <b>Confidence:</b> {event.get("confidence", 0):.2%}
                </div>
                """,
                unsafe_allow_html=True
            )


    # -----------------------------------------------------
    # Event table
    # -----------------------------------------------------
    if events:

        st.markdown(
            '<div class="section-title">📋 Event Details</div>',
            unsafe_allow_html=True
        )

        table_data = []

        for event in events:

            table_data.append({
                "Event ID": event.get("event_id"),
                "Track ID": event.get("track_id"),
                "Missing PPE": ", ".join(
                    event.get("missing_ppe", [])
                ),
                "Severity": event.get("severity"),
                "Confidence": f'{event.get("confidence", 0):.2%}',
                "Video Time": f'{event.get("video_time_s", 0):.2f}s'
            })

        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )


    # -----------------------------------------------------
    # JSON result
    # -----------------------------------------------------
    results_file = output_dir / "results.json"

    if results_file.exists():

        st.markdown(
            '<div class="section-title">📄 Analysis Report</div>',
            unsafe_allow_html=True
        )

        with open(results_file, "rb") as json_file:

            st.download_button(
                label="⬇️ Download Results JSON",
                data=json_file,
                file_name="NEXORA_PPE_results.json",
                mime="application/json"
            )

        with st.expander("🔍 View Raw JSON"):

            st.json(result)


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")

st.markdown(
    """
    <div style="text-align:center;color:#607D8B;padding:10px;">
        🛡️ <b>NEXORA Module 05</b> • PPE & Workplace Safety Monitoring
        <br>
        AI-based Helmet & Safety Vest Compliance
    </div>
    """,
    unsafe_allow_html=True
)
