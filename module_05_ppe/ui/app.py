import sys
import subprocess
from pathlib import Path

import streamlit as st


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from module_05_ppe.inference.pipeline import process_video


# =========================================================
# TRAINED MODEL
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "module_05_ppe"
    / "models"
    / "training_runs"
    / "ppe_v1"
    / "weights"
    / "best.pt"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="NEXORA - PPE Safety",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

/* Main application background */
.stApp {
    background-color: #F4FAFF;
}

/* Page spacing */
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

/* Text input */
.stTextInput > div > div > input {
    border-radius: 8px;
    border: 1px solid #90CAF9;
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

/* Status */
.status-online {
    color: #2E7D32;
    font-weight: 700;
}

.status-warning {
    color: #F57C00;
    font-weight: 700;
}

/* Custom section title */
.section-title {
    color: #1565C0;
    font-size: 22px;
    font-weight: 700;
    margin-top: 25px;
    margin-bottom: 12px;
}

</style>
""",
    unsafe_allow_html=True
)


# =========================================================
# FUNCTION: CONVERT VIDEO TO H264
# =========================================================

def convert_to_browser_video(input_video, output_video):

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_video),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_video)
    ]

    try:

        subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        return output_video

    except FileNotFoundError:

        raise RuntimeError(
            "FFmpeg was not found. "
            "Please install it using: brew install ffmpeg"
        )

    except subprocess.CalledProcessError as e:

        error_message = e.stderr.decode(
            "utf-8",
            errors="ignore"
        )

        raise RuntimeError(
            f"FFmpeg conversion failed:\n{error_message}"
        )


# =========================================================
# SESSION STATE
# =========================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "output_dir" not in st.session_state:
    st.session_state.output_dir = None


# =========================================================
# HEADER
# =========================================================

st.html(
    """
    <div style="
        background: linear-gradient(135deg, #E3F2FD, #F4FAFF);
        border: 1px solid #BBDEFB;
        border-radius: 18px;
        padding: 25px 30px;
        margin-bottom: 25px;
    ">

        <div style="
            font-size: 34px;
            font-weight: 800;
            color: #0D47A1;
        ">
            🛡️ NEXORA PPE SAFETY MONITORING
        </div>

        <div style="
            color: #546E7A;
            font-size: 16px;
            margin-top: 8px;
        ">
            AI-powered workplace safety monitoring •
            Helmet & Safety Vest Detection
        </div>

    </div>
    """
)


# =========================================================
# SIDEBAR
# =========================================================

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
        type=[
            "mp4",
            "mov",
            "avi",
            "mkv"
        ]
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

    # Model status

    if MODEL_PATH.exists():

        st.html(
            """
            <div style="
                color:#2E7D32;
                font-weight:700;
            ">
                ● Detection Engine Ready
            </div>
            """
        )

    else:

        st.html(
            """
            <div style="
                color:#F57C00;
                font-weight:700;
            ">
                ⚠ Trained model not found
            </div>
            """
        )


# =========================================================
# VIDEO ANALYSIS
# =========================================================

st.html(
    """
    <div style="
        color:#1565C0;
        font-size:22px;
        font-weight:700;
        margin-top:25px;
        margin-bottom:12px;
    ">
        📹 Video Analysis
    </div>
    """
)


# =========================================================
# NO VIDEO
# =========================================================

if uploaded_video is None:

    st.info(
        "Upload a workplace CCTV/video file from the "
        "left panel to start PPE monitoring."
    )


# =========================================================
# VIDEO UPLOADED
# =========================================================

else:

    st.success(
        f"Video loaded: **{uploaded_video.name}**"
    )

    col1, col2 = st.columns(
        [3, 1]
    )


    # =====================================================
    # INPUT VIDEO
    # =====================================================

    with col1:

        st.subheader("🎥 Input Video")

        st.video(
            uploaded_video
        )


    # =====================================================
    # ANALYSIS CONTROL
    # =====================================================

    with col2:

        st.subheader("🎬 Analysis")

        st.write(
            f"**Camera:** {camera_id}"
        )

        st.write(
            f"**File:** {uploaded_video.name}"
        )

        st.write(
            "**Model:** YOLO11s"
        )

        if quick_test:

            st.info(
                "Quick test: first 300 frames"
            )

        else:

            st.info(
                "Full video processing"
            )


        # =================================================
        # START DETECTION
        # =================================================

        if st.button(
            "🚀 Start PPE Detection",
            use_container_width=True
        ):

            # ---------------------------------------------
            # CHECK MODEL
            # ---------------------------------------------

            if not MODEL_PATH.exists():

                st.error(
                    "Trained model was not found."
                )

                st.code(
                    str(MODEL_PATH)
                )

                st.stop()


            # ---------------------------------------------
            # SAVE UPLOADED VIDEO
            # ---------------------------------------------

            temp_dir = (
                PROJECT_ROOT /
                "videos"
            )

            temp_dir.mkdir(
                exist_ok=True
            )

            input_path = (
                temp_dir /
                uploaded_video.name
            )


            with open(
                input_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_video.getbuffer()
                )


            # ---------------------------------------------
            # RUN DETECTION
            # ---------------------------------------------

            try:

                with st.spinner(
                    "🤖 AI is analyzing the video... Please wait."
                ):

                    result = process_video(

                        source=str(
                            input_path
                        ),

                        camera_id=camera_id,

                        save_annotated=True,

                        max_frames=max_frames,

                        weights_path=str(
                            MODEL_PATH
                        )
                    )


                # -----------------------------------------
                # SAVE RESULT IN SESSION
                # -----------------------------------------

                st.session_state.result = result

                st.session_state.output_dir = Path(
                    result["files"]["output_dir"]
                )


                st.success(
                    "✅ Video analysis completed!"
                )


                # Refresh page

                st.rerun()


            except Exception as e:

                st.error(
                    "❌ Detection failed."
                )

                st.exception(e)


# =========================================================
# RESULTS
# =========================================================

result = st.session_state.result


if result is not None:

    events = result.get(
        "events",
        []
    )


    # =====================================================
    # SAFETY OVERVIEW
    # =====================================================

    st.html(
        """
        <div style="
            color:#1565C0;
            font-size:22px;
            font-weight:700;
            margin-top:25px;
            margin-bottom:12px;
        ">
            📊 Safety Overview
        </div>
        """
    )


    # =====================================================
    # EVENT COUNTS
    # =====================================================

    total_events = len(
        events
    )


    critical_events = sum(
        1
        for e in events
        if e.get("severity") == "CRITICAL"
    )


    high_events = sum(
        1
        for e in events
        if e.get("severity") == "HIGH"
    )


    medium_events = sum(
        1
        for e in events
        if e.get("severity") == "MEDIUM"
    )


    # =====================================================
    # METRICS
    # =====================================================

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


    # =====================================================
    # PROCESSING INFORMATION
    # =====================================================

    st.html(
        """
        <div style="
            color:#1565C0;
            font-size:22px;
            font-weight:700;
            margin-top:25px;
            margin-bottom:12px;
        ">
            📈 Processing Information
        </div>
        """
    )


    col1, col2, col3, col4 = st.columns(4)


    # Camera

    with col1:

        st.metric(
            "Camera",
            result.get(
                "camera_id",
                camera_id
            )
        )


    # Frames

    with col2:

        st.metric(
            "Frames",
            result.get(
                "frames_processed",
                0
            )
        )


    # FPS

    with col3:

        st.metric(
            "FPS",
            result.get(
                "video_fps",
                0
            )
        )


    # Processing time

    with col4:

        processing = result.get(
            "processing",
            {}
        )


        if isinstance(
            processing,
            dict
        ):

            processing_time = processing.get(
                "wall_seconds",
                processing.get(
                    "elapsed_seconds",
                    processing.get(
                        "seconds",
                        0
                    )
                )
            )

        else:

            processing_time = processing


        if isinstance(
            processing_time,
            (int, float)
        ):

            processing_display = (
                f"{processing_time:.2f}s"
            )

        else:

            processing_display = str(
                processing_time
            )


        st.metric(
            "Processing",
            processing_display
        )


    # =====================================================
    # ANNOTATED VIDEO
    # =====================================================

    output_dir = (
        st.session_state.output_dir
    )


    if output_dir is not None:

        annotated_video = (
            output_dir /
            "annotated_video.mp4"
        )


        # =================================================
        # VIDEO EXISTS
        # =================================================

        if annotated_video.exists():

            st.html(
                """
                <div style="
                    color:#1565C0;
                    font-size:22px;
                    font-weight:700;
                    margin-top:25px;
                    margin-bottom:12px;
                ">
                    🎥 Detection Result
                </div>
                """
            )


            # ---------------------------------------------
            # H264 OUTPUT
            # ---------------------------------------------

            browser_video = (
                output_dir /
                "annotated_video_h264.mp4"
            )


            try:

                # Convert to H264

                if not browser_video.exists():

                    with st.spinner(
                        "🎞️ Preparing detection video..."
                    ):

                        convert_to_browser_video(
                            annotated_video,
                            browser_video
                        )


                # -----------------------------------------
                # DISPLAY VIDEO
                # -----------------------------------------

                if browser_video.exists():

                    st.success(
                        "✅ Annotated PPE detection video ready."
                    )


                    with open(
                        browser_video,
                        "rb"
                    ) as video_file:

                        video_bytes = (
                            video_file.read()
                        )


                    st.video(
                        video_bytes
                    )


                    # -------------------------------------
                    # DOWNLOAD VIDEO
                    # -------------------------------------

                    st.download_button(

                        label="⬇️ Download Annotated Video",

                        data=video_bytes,

                        file_name=(
                            "NEXORA_PPE_Annotated.mp4"
                        ),

                        mime="video/mp4",

                        use_container_width=True
                    )


            except Exception as e:

                st.error(
                    "❌ Could not prepare annotated video."
                )

                st.exception(e)


        # =================================================
        # VIDEO DOES NOT EXIST
        # =================================================

        else:

            st.warning(
                "⚠️ Detection completed, but "
                "annotated_video.mp4 was not created."
            )

            st.write(
                "**Output folder:**"
            )

            st.code(
                str(output_dir)
            )

            st.info(
                "The detection results are still available "
                "below."
            )


    # =====================================================
    # DETECTED SAFETY EVENTS
    # =====================================================

    st.html(
        """
        <div style="
            color:#1565C0;
            font-size:22px;
            font-weight:700;
            margin-top:25px;
            margin-bottom:12px;
        ">
            🚨 Detected Safety Events
        </div>
        """
    )


    # =====================================================
    # NO EVENTS
    # =====================================================

    if len(events) == 0:

        st.success(
            "✅ No PPE violations detected in this video."
        )


    # =====================================================
    # EVENTS FOUND
    # =====================================================

    else:

        for event in events:

            severity = event.get(
                "severity",
                "UNKNOWN"
            )


            # ---------------------------------------------
            # SEVERITY
            # ---------------------------------------------

            if severity == "CRITICAL":

                background = "#FFEBEE"
                border = "#D32F2F"
                icon = "🔴"

            elif severity == "HIGH":

                background = "#FFF3E0"
                border = "#F57C00"
                icon = "🟠"

            else:

                background = "#E3F2FD"
                border = "#1976D2"
                icon = "🔵"


            # ---------------------------------------------
            # EVENT INFORMATION
            # ---------------------------------------------

            event_id = event.get(
                "event_id",
                "Event"
            )

            track_id = event.get(
                "track_id",
                "N/A"
            )

            missing = event.get(
                "missing_ppe",
                []
            )

            missing_text = ", ".join(
                str(x)
                .replace(
                    "_",
                    " "
                )
                .title()
                for x in missing
            )

            confidence = event.get(
                "confidence",
                0
            )


            # ---------------------------------------------
            # EVENT CARD
            # ---------------------------------------------

            st.html(
                f"""
                <div style="
                    background:{background};
                    border-left:5px solid {border};
                    border-radius:14px;
                    padding:18px;
                    margin:10px 0;
                    box-shadow:0 3px 10px
                    rgba(25,118,210,0.08);
                ">

                    <div style="
                        color:{border};
                        font-size:20px;
                        font-weight:700;
                        margin-bottom:10px;
                    ">
                        {icon} {severity} — {event_id}
                    </div>

                    <div style="
                        color:#37474F;
                        line-height:1.8;
                    ">

                        <b>Track:</b>
                        {track_id}

                        <br>

                        <b>Missing PPE:</b>
                        {missing_text}

                        <br>

                        <b>Confidence:</b>
                        {confidence:.2%}

                    </div>

                </div>
                """
            )


    # =====================================================
    # EVENT TABLE
    # =====================================================

    if events:

        st.html(
            """
            <div style="
                color:#1565C0;
                font-size:22px;
                font-weight:700;
                margin-top:25px;
                margin-bottom:12px;
            ">
                📋 Event Details
            </div>
            """
        )


        table_data = []


        for event in events:

            table_data.append({

                "Event ID":
                    event.get(
                        "event_id"
                    ),

                "Track ID":
                    event.get(
                        "track_id"
                    ),

                "Missing PPE":
                    ", ".join(
                        event.get(
                            "missing_ppe",
                            []
                        )
                    ),

                "Severity":
                    event.get(
                        "severity"
                    ),

                "Confidence":
                    f'{event.get("confidence", 0):.2%}',

                "Video Time":
                    f'{event.get("video_time_s", 0):.2f}s'
            })


        st.dataframe(
            table_data,
            use_container_width=True,
            hide_index=True
        )


    # =====================================================
    # ANALYSIS REPORT
    # =====================================================

    results_file = (
        output_dir /
        "results.json"
    )


    if results_file.exists():

        st.html(
            """
            <div style="
                color:#1565C0;
                font-size:22px;
                font-weight:700;
                margin-top:25px;
                margin-bottom:12px;
            ">
                📄 Analysis Report
            </div>
            """
        )


        with open(
            results_file,
            "rb"
        ) as json_file:

            json_bytes = (
                json_file.read()
            )


        st.download_button(

            label="⬇️ Download Results JSON",

            data=json_bytes,

            file_name="NEXORA_PPE_results.json",

            mime="application/json"
        )


        # Raw JSON

        with st.expander(
            "🔍 View Raw JSON"
        ):

            st.json(
                result
            )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.html(
    """
    <div style="
        text-align:center;
        color:#607D8B;
        padding:10px;
        font-size:15px;
    ">

        🛡️ <b>NEXORA Module 05</b>
        • PPE & Workplace Safety Monitoring

        <br><br>

        AI-based Helmet & Safety Vest Compliance

    </div>
    """
)
