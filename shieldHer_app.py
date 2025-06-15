import streamlit as st
import json
import time
import threading
import cv2
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import os
import queue

# Audio processing imports (with fallback handling)
try:
    import sounddevice as sd
    import librosa
    import tensorflow as tf
    import vosk
    import scipy.signal
    from pynput import keyboard
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    st.warning(f"Audio processing libraries not available: {e}")
    st.info("Install required packages: pip install sounddevice librosa tensorflow vosk pynput scipy")

# Configure Streamlit page
st.set_page_config(
    page_title="🛡️ ShieldHer - Personal Safety Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Audio Configuration
MODEL_PATH = "ai_model/voice_stress_model.tflite"
VOSK_MODEL_PATH = "vosk_model"
KEYWORDS = [
    "help", "emergency", "save me", "please help", "leave me", "stop", "no please", 
    "help me", "donot touch me", "i said no", "let me go", "get away from me", 
    "you are scaring me", "i need help", "why are you doing this", "please stop", 
    "call the police", "i donot feel safe", "this is not okay", "back off",
    "stop following me", "i am not okay", "you are hurting me", "donot come closer", 
    "stay away", "this is harassment", "i want to leave"
]
STRESS_THRESHOLD = 0.85
DURATION = 4
SAMPLE_RATE = 16000
BLOCKSIZE = 4000
MIN_AUDIO_LEVEL = 0.001

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    
    .emergency-card {
        background: linear-gradient(135deg, #ff6b6b, #ee5a24);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .safe-card {
        background: linear-gradient(135deg, #00d2d3, #54a0ff);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .audio-status {
        background: linear-gradient(135deg, #2ecc71, #27ae60);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-safe { background-color: #2ecc71; }
    .status-warning { background-color: #f39c12; }
    .status-emergency { background-color: #e74c3c; }
</style>
""", unsafe_allow_html=True)

class AudioMonitor:
    """Audio monitoring class integrated with Streamlit"""
    
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.is_monitoring = False
        self.interpreter = None
        self.vosk_model = None
        self.models_loaded = False
        self.current_transcript = ""
        self.current_stress_score = 0.0
        self.detected_keywords = []
        self.audio_level = 0.0
        self.setup_models()
    
    def setup_models(self):
        """Initialize ML models"""
        if not AUDIO_AVAILABLE:
            return
            
        try:
            # Setup TensorFlow Lite model
            if os.path.exists(MODEL_PATH):
                self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                st.session_state.tflite_loaded = True
            else:
                st.session_state.tflite_loaded = False
        except Exception as e:
            st.session_state.tflite_loaded = False
            st.error(f"TFLite model loading error: {e}")
        
        try:
            # Setup Vosk model
            if os.path.exists(VOSK_MODEL_PATH):
                self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
                st.session_state.vosk_loaded = True
            else:
                st.session_state.vosk_loaded = False
        except Exception as e:
            st.session_state.vosk_loaded = False
            st.error(f"Vosk model loading error: {e}")
        
        self.models_loaded = (self.interpreter is not None and self.vosk_model is not None)
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio callback for real-time processing"""
        if not self.is_monitoring:
            return
            
        rms = np.sqrt(np.mean(indata**2))
        max_val = np.max(np.abs(indata))
        
        self.audio_level = rms
        
        self.audio_queue.put({
            'data': indata.copy(),
            'rms': rms,
            'max': max_val,
            'timestamp': time.time()
        })
    
    def extract_features(self, audio_data):
        """Extract features from audio data for stress detection"""
        try:
            # Extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
            features = np.mean(mfccs.T, axis=0)
            return features.reshape(1, -1).astype(np.float32)
        except Exception as e:
            return np.zeros((1, 13), dtype=np.float32)
    
    def detect_stress(self, audio_data):
        """Detect stress using TensorFlow Lite model"""
        if self.interpreter is None:
            return 0.0
        
        try:
            features = self.extract_features(audio_data)
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            return float(output_data[0])
        except Exception as e:
            return 0.0
    
    def detect_keywords(self, audio_chunks_data):
        """Keyword detection using Vosk"""
        if self.vosk_model is None:
            return "", []
        
        try:
            recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            transcript = ""
            
            for chunk_info in audio_chunks_data:
                chunk = chunk_info['data'].flatten()
                
                if chunk_info['rms'] < MIN_AUDIO_LEVEL:
                    continue
                
                audio_int16 = (chunk * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                if recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        transcript += text + " "
            
            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                transcript += final_text
            
            transcript = transcript.strip().lower()
            detected_keywords = [kw for kw in KEYWORDS if kw.lower() in transcript]
            
            return transcript, detected_keywords
            
        except Exception as e:
            return "", []
    
    def start_monitoring(self):
        """Start audio monitoring in a separate thread"""
        if not AUDIO_AVAILABLE:
            st.error("Audio monitoring not available - missing dependencies")
            return
            
        self.is_monitoring = True
        
        def monitor_thread():
            try:
                with sd.InputStream(
                    callback=self.audio_callback,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCKSIZE,
                    dtype=np.float32,
                    channels=1
                ):
                    audio_buffer = []
                    buffer_start_time = time.time()
                    
                    while self.is_monitoring:
                        current_time = time.time()
                        
                        # Collect audio for DURATION seconds
                        if current_time - buffer_start_time < DURATION:
                            try:
                                chunk_info = self.audio_queue.get(timeout=0.1)
                                audio_buffer.append(chunk_info)
                            except queue.Empty:
                                continue
                        else:
                            # Process collected audio
                            if audio_buffer:
                                # Combine audio data
                                combined_audio = np.concatenate([chunk['data'].flatten() for chunk in audio_buffer])
                                
                                # Check if not silent
                                rms = np.sqrt(np.mean(combined_audio**2))
                                if rms > MIN_AUDIO_LEVEL:
                                    # Stress detection
                                    if self.interpreter is not None:
                                        self.current_stress_score = self.detect_stress(combined_audio)
                                    
                                    # Keyword detection
                                    if self.vosk_model is not None:
                                        self.current_transcript, self.detected_keywords = self.detect_keywords(audio_buffer)
                                    
                                    # Update session state for real-time display
                                    st.session_state.sensor_data["stress_score"] = self.current_stress_score
                                    if self.detected_keywords:
                                        st.session_state.sensor_data["keyword"] = True
                                        st.session_state.sensor_data["emotion"] = "fear"
                                    
                                    # Emergency detection
                                    if (self.current_stress_score >= STRESS_THRESHOLD or 
                                        len(self.detected_keywords) > 0):
                                        # Trigger emergency alert
                                        alert_data = create_sample_alert_data()
                                        alert_data["stress_score"] = self.current_stress_score
                                        alert_data["keywords"] = self.detected_keywords
                                        alert_data["transcript"] = self.current_transcript
                                        save_alert_data(alert_data)
                                        st.session_state.alerts_history.append(alert_data)
                            
                            # Reset for next cycle
                            audio_buffer = []
                            buffer_start_time = current_time
                            
                            # Clear queue
                            while not self.audio_queue.empty():
                                try:
                                    self.audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                            
                            time.sleep(0.5)
                            
            except Exception as e:
                st.error(f"Audio monitoring error: {e}")
    
        # Start monitoring thread
        monitor_thread = threading.Thread(target=monitor_thread, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop audio monitoring"""
        self.is_monitoring = False

# Initialize session state
if 'sensor_data' not in st.session_state:
    st.session_state.sensor_data = {
        "emotion": "neutral",
        "keyword": False,
        "fall": False,
        "stress_score": 0.0,
        "location": {"lat": 28.6139, "lon": 77.209}
    }

if 'event_log' not in st.session_state:
    st.session_state.event_log = []

if 'alerts_history' not in st.session_state:
    st.session_state.alerts_history = []

if 'monitoring_active' not in st.session_state:
    st.session_state.monitoring_active = False

if 'audio_monitoring_active' not in st.session_state:
    st.session_state.audio_monitoring_active = False

if 'contacts' not in st.session_state:
    st.session_state.contacts = ["Emergency Contact 1", "Emergency Contact 2"]

if 'audio_monitor' not in st.session_state:
    st.session_state.audio_monitor = AudioMonitor()

if 'tflite_loaded' not in st.session_state:
    st.session_state.tflite_loaded = False

if 'vosk_loaded' not in st.session_state:
    st.session_state.vosk_loaded = False

# Helper Functions
def create_sample_alert_data():
    """Create sample alert data for demonstration"""
    return {
        "status": "emergency" if st.session_state.sensor_data["keyword"] or 
                 st.session_state.sensor_data["emotion"] in ["fear", "angry"] else "safe",
        "timestamp": datetime.now().isoformat(),
        "location": st.session_state.sensor_data["location"],
        "stress_score": st.session_state.sensor_data["stress_score"],
        "keywords": st.session_state.audio_monitor.detected_keywords if hasattr(st.session_state.audio_monitor, 'detected_keywords') else [],
        "transcript": st.session_state.audio_monitor.current_transcript if hasattr(st.session_state.audio_monitor, 'current_transcript') else "No audio detected",
        "recording": "recordings/alert_sample.wav",
        "contacts_notified": st.session_state.contacts if st.session_state.sensor_data["keyword"] else [],
        "emotion": st.session_state.sensor_data["emotion"]
    }

def save_alert_data(alert_data):
    """Save alert data to JSON file"""
    os.makedirs("data", exist_ok=True)
    with open("data/alert_data.json", "w") as f:
        json.dump(alert_data, f, indent=2)

def load_alert_data():
    """Load alert data from JSON file"""
    try:
        with open("data/alert_data.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return create_sample_alert_data()

def risk_assessment():
    """Calculate risk level based on current data"""
    risk_score = 0
    
    # Emotion-based risk
    if st.session_state.sensor_data["emotion"] in ["fear", "angry"]:
        risk_score += 0.4
    elif st.session_state.sensor_data["emotion"] in ["sad", "disgust"]:
        risk_score += 0.2
    
    # Keyword detection
    if st.session_state.sensor_data["keyword"]:
        risk_score += 0.4
    
    # Fall detection
    if st.session_state.sensor_data["fall"]:
        risk_score += 0.3
    
    # Stress score
    risk_score += st.session_state.sensor_data["stress_score"] * 0.3
    
    return min(risk_score, 1.0)

def get_risk_level(risk_score):
    """Get risk level category"""
    if risk_score >= 0.7:
        return "HIGH", "🔴"
    elif risk_score >= 0.4:
        return "MEDIUM", "🟡"
    else:
        return "LOW", "🟢"

# Main App Layout
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🛡️ ShieldHer - Personal Safety Monitor</h1>
        <p>Real-time emotion, voice, and safety monitoring system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar Navigation
    with st.sidebar:
        st.image("https://via.placeholder.com/200x100/667eea/white?text=ShieldHer", width=200)
        
        page = st.selectbox(
            "Navigate",
            ["🏠 Dashboard", "🎤 Audio Monitor", "📊 Analytics", "🚨 Alerts", "⚙️ Settings", "📱 Contacts"]
        )
        
        st.markdown("---")
        
        # Audio System Status
        st.markdown("### 🎤 Audio System")
        if AUDIO_AVAILABLE:
            tflite_status = "✅" if st.session_state.tflite_loaded else "❌"
            vosk_status = "✅" if st.session_state.vosk_loaded else "❌"
            st.markdown(f"{tflite_status} Stress Detection Model")
            st.markdown(f"{vosk_status} Voice Recognition Model")
        else:
            st.markdown("❌ Audio libraries not installed")
        
        # Monitoring Control
        st.markdown("### 🎛️ Monitoring Control")
        
        # Visual monitoring toggle
        if st.button("🔴 Start Visual Monitoring" if not st.session_state.monitoring_active else "⏸️ Stop Visual Monitoring"):
            st.session_state.monitoring_active = not st.session_state.monitoring_active
            if st.session_state.monitoring_active:
                st.success("Visual Monitoring Started!")
            else:
                st.info("Visual Monitoring Stopped!")
        
        # Audio monitoring toggle
        if AUDIO_AVAILABLE and (st.session_state.tflite_loaded or st.session_state.vosk_loaded):
            if st.button("🎤 Start Audio Monitoring" if not st.session_state.audio_monitoring_active else "🔇 Stop Audio Monitoring"):
                st.session_state.audio_monitoring_active = not st.session_state.audio_monitoring_active
                if st.session_state.audio_monitoring_active:
                    st.session_state.audio_monitor.start_monitoring()
                    st.success("Audio Monitoring Started!")
                else:
                    st.session_state.audio_monitor.stop_monitoring()
                    st.info("Audio Monitoring Stopped!")
        else:
            st.button("🎤 Audio Monitoring (Unavailable)", disabled=True)
            if AUDIO_AVAILABLE:
                st.caption("Models not loaded")
            else:
                st.caption("Audio libraries missing")
        
        # Emergency Button
        st.markdown("---")
        if st.button("🚨 EMERGENCY ALERT", type="primary"):
            st.session_state.sensor_data["keyword"] = True
            st.session_state.sensor_data["stress_score"] = 0.95
            alert_data = create_sample_alert_data()
            save_alert_data(alert_data)
            st.session_state.alerts_history.append(alert_data)
            st.error("🚨 EMERGENCY ALERT TRIGGERED!")
            st.balloons()
        
        # System Status
        st.markdown("---")
        st.markdown("### 📡 System Status")
        visual_status = "🟢" if st.session_state.monitoring_active else "🔴"
        audio_status = "🟢" if st.session_state.audio_monitoring_active else "🔴"
        st.markdown(f"{visual_status} **Visual**: {'Active' if st.session_state.monitoring_active else 'Inactive'}")
        st.markdown(f"{audio_status} **Audio**: {'Active' if st.session_state.audio_monitoring_active else 'Inactive'}")
        
        # Audio level indicator
        if hasattr(st.session_state.audio_monitor, 'audio_level'):
            audio_level = st.session_state.audio_monitor.audio_level
            st.markdown(f"🔊 **Audio Level**: {audio_level:.4f}")
            st.progress(min(audio_level * 1000, 1.0))
        
        # Current Risk Level
        risk_score = risk_assessment()
        risk_level, risk_emoji = get_risk_level(risk_score)
        st.markdown(f"{risk_emoji} **Risk Level**: {risk_level}")
        st.progress(risk_score)
    
    # Main Content Area
    if page == "🏠 Dashboard":
        dashboard_page()
    elif page == "🎤 Audio Monitor":
        audio_monitor_page()
    elif page == "📊 Analytics":
        analytics_page()
    elif page == "🚨 Alerts":
        alerts_page()
    elif page == "⚙️ Settings":
        settings_page()
    elif page == "📱 Contacts":
        contacts_page()

def dashboard_page():
    """Main dashboard page"""
    
    # Real-time Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        current_emotion = st.session_state.sensor_data["emotion"]
        emotion_emoji = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨", "neutral": "😐"}.get(current_emotion, "😐")
        st.metric(
            label="Current Emotion",
            value=f"{emotion_emoji} {current_emotion.title()}",
            delta="Normal" if current_emotion in ["happy", "neutral"] else "Alert"
        )
    
    with col2:
        stress_score = st.session_state.sensor_data["stress_score"]
        st.metric(
            label="Stress Level",
            value=f"{stress_score:.1%}",
            delta="High" if stress_score > 0.7 else "Normal"
        )
    
    with col3:
        keyword_status = "Detected" if st.session_state.sensor_data["keyword"] else "None"
        st.metric(
            label="Voice Keywords",
            value=keyword_status,
            delta="Alert" if st.session_state.sensor_data["keyword"] else "Clear"
        )
    
    with col4:
        location = st.session_state.sensor_data["location"]
        st.metric(
            label="Location Status",
            value="Tracked",
        )
    
    st.markdown("---")
    
    # Live Data Visualization
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Real-time Monitoring")
        
        # Create sample time series data
        if 'time_series_data' not in st.session_state:
            st.session_state.time_series_data = []
        
        # Add current data point
        current_time = datetime.now()
        st.session_state.time_series_data.append({
            'time': current_time,
            'stress_score': st.session_state.sensor_data["stress_score"],
            'risk_level': risk_assessment(),
            'audio_level': getattr(st.session_state.audio_monitor, 'audio_level', 0.0)
        })
        
        # Keep only last 50 points
        if len(st.session_state.time_series_data) > 50:
            st.session_state.time_series_data = st.session_state.time_series_data[-50:]
        
        if st.session_state.time_series_data:
            df = pd.DataFrame(st.session_state.time_series_data)
            
            fig = make_subplots(
                rows=3, cols=1,
                subplot_titles=('Stress Level', 'Risk Assessment', 'Audio Level'),
                vertical_spacing=0.1
            )
            
            # Stress level line
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['stress_score'],
                    mode='lines+markers',
                    name='Stress Level',
                    line=dict(color='#ff6b6b', width=2)
                ),
                row=1, col=1
            )
            
            # Risk level line
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['risk_level'],
                    mode='lines+markers',
                    name='Risk Level',
                    line=dict(color='#4ecdc4', width=2)
                ),
                row=2, col=1
            )
            
            # Audio level line
            fig.add_trace(
                go.Scatter(
                    x=df['time'],
                    y=df['audio_level'],
                    mode='lines+markers',
                    name='Audio Level',
                    line=dict(color='#f39c12', width=2)
                ),
                row=3, col=1
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                margin=dict(l=0, r=0, t=30, b=0)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Current Status")
        
        # Status indicator
        alert_data = load_alert_data()
        status = alert_data.get("status", "safe")
        
        if status == "emergency":
            st.markdown("""
            <div class="emergency-card">
                <h3>🚨 EMERGENCY</h3>
                <p>Immediate attention required!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="safe-card">
                <h3>✅ SAFE</h3>
                <p>All systems normal</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Recent transcript
        st.markdown("### 🗣️ Recent Audio")
        transcript = getattr(st.session_state.audio_monitor, 'current_transcript', "No recent audio")
        st.text_area("Last detected speech:", transcript, height=100)
        
        # Keywords detected
        keywords = getattr(st.session_state.audio_monitor, 'detected_keywords', [])
        if keywords:
            st.markdown("**🔑 Keywords detected:**")
            for keyword in keywords:
                st.markdown(f"- `{keyword}`")
    
    # Quick Actions
    st.markdown("---")
    st.markdown("### ⚡ Quick Actions")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🧠 Simulate Fear", key="fear"):
            st.session_state.sensor_data["emotion"] = "fear"
            st.session_state.sensor_data["stress_score"] = 0.8
            st.rerun()
    
    with col2:
        if st.button("😠 Simulate Anger", key="anger"):
            st.session_state.sensor_data["emotion"] = "angry"
            st.session_state.sensor_data["stress_score"] = 0.7
            st.rerun()
    
    with col3:
        if st.button("🔊 Trigger Voice Alert", key="voice"):
            st.session_state.sensor_data["keyword"] = True
            st.rerun()
    
    with col4:
        if st.button("😊 Reset to Normal", key="reset"):
            st.session_state.sensor_data = {
                "emotion": "neutral",
                "keyword": False,
                "fall": False,
                "stress_score": 0.1,
                "location": {"lat": 28.6139, "lon": 77.209}
            }
            st.rerun()

def audio_monitor_page():
    """Audio monitoring page with detailed controls and diagnostics"""
    st.markdown("## 🎤 Audio Monitor")
    
    if not AUDIO_AVAILABLE:
        st.error("Audio monitoring is not available. Please install required packages:")
        st.code("pip install sounddevice librosa tensorflow vosk pynput scipy")
        return
    
    # System Status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🧠 Stress Detection")
        if st.session_state.tflite_loaded:
            st.success("✅ Model Loaded")
            current_stress = getattr(st.session_state.audio_monitor, 'current_stress_score', 0.0)
            st.metric("Current Stress", f"{current_stress:.1%}")
        else:
            st.error("❌ Model Not Found")
            st.info(f"Expected path: {MODEL_PATH}")
    
    with col2:
        st.markdown("### 🗣️ Voice Recognition")
        if st.session_state.vosk_loaded:
            st.success("✅ Model Loaded")
            transcript = getattr(st.session_state.audio_monitor, 'current_transcript', "")
            st.text_area("Recent Transcript", transcript, height=100)
        else:
            st.error("❌ Model Not Found")
            st.info(f"Expected path: {VOSK_MODEL_PATH}")
    
    with col3:
        st.markdown("### 🔊 Audio Level")
        audio_level = getattr(st.session_state.audio_monitor, 'audio_level', 0.0)
        st.metric("Current Level", f"{audio_level:.6f}")
        st.progress(min(audio_level * 1000, 1.0))
    
    # Keywords Detection
    st.markdown("---")
    st.markdown("### 🔑 Emergency Keywords Detection")
    
    keywords = getattr(st.session_state.audio_monitor, 'detected_keywords', [])
    if keywords:
        st.error(f"⚠️ Emergency keywords detected: {', '.join(keywords)}")
    else:
        st.success("✅ No emergency keywords detected")
    
    # Monitored Keywords List
    with st.expander("View All Monitored Keywords"):
        st.write("The system monitors for these emergency keywords:")
        for i, keyword in enumerate(st.session_state.audio_monitor.emergency_keywords, 1):
        st.write(f"{i}. **{keyword}**")
    
    # Add custom keyword functionality
    st.markdown("---")
    st.markdown("**Add Custom Keywords:**")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        new_keyword = st.text_input("Enter new emergency keyword:", placeholder="e.g., urgent, crisis")
    with col2:
        if st.button("Add Keyword", type="secondary"):
            if new_keyword and new_keyword.lower().strip():
                keyword_to_add = new_keyword.lower().strip()
                if keyword_to_add not in st.session_state.audio_monitor.emergency_keywords:
                    st.session_state.audio_monitor.emergency_keywords.append(keyword_to_add)
                    st.success(f"Added '{keyword_to_add}' to monitored keywords!")
                    st.rerun()
                else:
                    st.warning("This keyword is already being monitored.")
    
    # Remove keyword functionality
    if st.session_state.audio_monitor.emergency_keywords:
        st.markdown("**Remove Keywords:**")
        keyword_to_remove = st.selectbox(
            "Select keyword to remove:",
            options=[""] + st.session_state.audio_monitor.emergency_keywords,
            format_func=lambda x: "Select a keyword..." if x == "" else x
        )
        
        if keyword_to_remove and st.button("Remove Keyword", type="secondary"):
            st.session_state.audio_monitor.emergency_keywords.remove(keyword_to_remove)
            st.success(f"Removed '{keyword_to_remove}' from monitored keywords!")
            st.rerun()

# Detection History
st.markdown("---")
st.markdown("### 📊 Detection History")

if hasattr(st.session_state.audio_monitor, 'detection_history') and st.session_state.audio_monitor.detection_history:
    history_df = pd.DataFrame(st.session_state.audio_monitor.detection_history)
    history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
    
    # Display recent detections
    st.dataframe(
        history_df.sort_values('timestamp', ascending=False).head(10),
        use_container_width=True,
        column_config={
            "timestamp": st.column_config.DatetimeColumn(
                "Detection Time",
                format="MM/DD/YY HH:mm:ss"
            ),
            "keyword": st.column_config.TextColumn("Keyword Detected"),
            "confidence": st.column_config.ProgressColumn(
                "Confidence",
                min_value=0,
                max_value=1
            )
        }
    )
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Detections", len(history_df))
    with col2:
        st.metric("Unique Keywords", history_df['keyword'].nunique())
    with col3:
        avg_confidence = history_df['confidence'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2%}")
        
else:
    st.info("No detection history available yet.")