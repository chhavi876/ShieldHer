import threading
import time
import json
import os
import numpy as np
from datetime import datetime, timezone
import sounddevice as sd
import speech_recognition as sr
import logging
from collections import deque
import re
import statistics

# Import your existing trigger system
try:
    from trigger_combined import check_and_trigger_emergency, EmergencyTrigger
    print("✅ Emergency trigger system imported successfully")
except ImportError as e:
    print(f"❌ Failed to import trigger_combined.py: {e}")
    print("Please ensure trigger_combined.py is in the same directory")
    exit(1)

# === CONFIGURATION ===
MONITORING_CONFIG = {
    'stress_threshold_high': 0.85,      # High stress threshold
    'stress_threshold_critical': 0.90,  # Critical stress threshold
    'audio_check_interval': 3,          # seconds between audio analysis
    'stress_window_size': 10,           # number of readings to average
    'keyword_sensitivity': 0.7,         # keyword detection sensitivity
    'emergency_cooldown': 300,          # 5 minutes cooldown between emergencies
    'continuous_monitoring': True,       # Enable continuous monitoring
    'log_level': 'INFO'                 # DEBUG, INFO, WARNING, ERROR
}

# Emergency keywords (you can expand this list)
EMERGENCY_KEYWORDS = {
    'high_priority': [
        'help', 'help me', 'emergency', 'save me', 'please help',
        'call police', 'call 911', 'call 100', 'i need help',
        'someone help me', 'get away', 'stop it', 'leave me alone',
        'dont touch me', "don't touch me", 'let me go', 'no no no'
    ],
    'medium_priority': [
        'scared', 'afraid', 'panic', 'stressed', 'harassment',
        'uncomfortable', 'inappropriate', 'wrong', 'stop',
        'quit it', 'back off', 'go away', 'not okay'
    ],
    'context_keywords': [
        'boss', 'manager', 'colleague', 'coworker', 'meeting',
        'office', 'workplace', 'work', 'job', 'professional'
    ]
}

class StressAnalyzer:
    """Simulated stress analyzer - replace with your actual stress detection model"""
    
    def __init__(self):
        self.baseline_established = False
        self.baseline_values = deque(maxlen=50)
        self.recent_readings = deque(maxlen=MONITORING_CONFIG['stress_window_size'])
        
    def analyze_audio_stress(self, audio_data):
        """
        Analyze audio for stress indicators
        Replace this with your actual stress detection algorithm
        """
        try:
            # Simulated stress analysis based on audio characteristics
            # In reality, this would use ML models to analyze voice stress
            
            # Calculate basic audio features
            amplitude = np.abs(audio_data).mean()
            frequency_variance = np.var(np.fft.fft(audio_data))
            
            # Simulate stress calculation (replace with your model)
            stress_score = min(1.0, (amplitude * 2.0 + frequency_variance * 0.0001))
            
            # Add some realistic variation
            stress_score += np.random.normal(0, 0.05)
            stress_score = max(0.0, min(1.0, stress_score))
            
            # Store reading
            self.recent_readings.append(stress_score)
            
            # Calculate smoothed stress level
            if len(self.recent_readings) >= 3:
                smoothed_stress = statistics.mean(list(self.recent_readings)[-5:])
            else:
                smoothed_stress = stress_score
            
            return {
                'raw_stress': stress_score,
                'smoothed_stress': smoothed_stress,
                'confidence': min(1.0, len(self.recent_readings) / 10.0),
                'trend': self._calculate_trend()
            }
            
        except Exception as e:
            print(f"❌ Stress analysis error: {e}")
            return {'raw_stress': 0.0, 'smoothed_stress': 0.0, 'confidence': 0.0, 'trend': 'stable'}
    
    def _calculate_trend(self):
        """Calculate stress trend"""
        if len(self.recent_readings) < 5:
            return 'stable'
        
        recent = list(self.recent_readings)[-5:]
        if recent[-1] > recent[0] + 0.1:
            return 'increasing'
        elif recent[-1] < recent[0] - 0.1:
            return 'decreasing'
        else:
            return 'stable'

class AudioTranscriber:
    """Real-time audio transcription"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Calibrate microphone
        print("🎙️ Calibrating microphone...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
        print("✅ Microphone calibrated")
    
    def transcribe_audio(self, audio_data, sample_rate=16000):
        """Transcribe audio data to text"""
        try:
            # Convert numpy array to AudioData
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            audio_sr = sr.AudioData(audio_bytes, sample_rate, 2)
            
            # Recognize speech
            text = self.recognizer.recognize_google(audio_sr, language='en-US')
            return text.lower()
            
        except sr.UnknownValueError:
            return ""  # No speech detected
        except sr.RequestError as e:
            print(f"⚠️ Speech recognition error: {e}")
            return ""
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return ""

class KeywordAnalyzer:
    """Analyze transcripts for emergency keywords"""
    
    def __init__(self):
        self.high_priority = EMERGENCY_KEYWORDS['high_priority']
        self.medium_priority = EMERGENCY_KEYWORDS['medium_priority']
        self.context_keywords = EMERGENCY_KEYWORDS['context_keywords']
    
    def analyze_keywords(self, transcript):
        """Analyze transcript for emergency keywords"""
        if not transcript:
            return {'detected_keywords': [], 'severity': 'none', 'confidence': 0.0}
        
        transcript = transcript.lower()
        detected_keywords = []
        severity_score = 0
        
        # Check high priority keywords
        for keyword in self.high_priority:
            if keyword in transcript:
                detected_keywords.append(keyword)
                severity_score += 3
        
        # Check medium priority keywords
        for keyword in self.medium_priority:
            if keyword in transcript:
                detected_keywords.append(keyword)
                severity_score += 1
        
        # Check context (workplace-related)
        workplace_context = any(keyword in transcript for keyword in self.context_keywords)
        if workplace_context and detected_keywords:
            severity_score *= 1.5  # Boost score for workplace context
        
        # Determine severity
        if severity_score >= 6:
            severity = 'critical'
        elif severity_score >= 3:
            severity = 'high'
        elif severity_score >= 1:
            severity = 'medium'
        else:
            severity = 'none'
        
        confidence = min(1.0, severity_score / 10.0)
        
        return {
            'detected_keywords': detected_keywords,
            'severity': severity,
            'confidence': confidence,
            'workplace_context': workplace_context,
            'raw_score': severity_score
        }

class EmergencyMonitor:
    """Main emergency monitoring system"""
    
    def __init__(self):
        self.stress_analyzer = StressAnalyzer()
        self.transcriber = AudioTranscriber()
        self.keyword_analyzer = KeywordAnalyzer()
        
        self.monitoring_active = False
        self.last_emergency_time = 0
        self.emergency_trigger = EmergencyTrigger()
        
        # Setup logging
        self.setup_logging()
        
        print("🛡️ Emergency Monitor initialized")
    
    def setup_logging(self):
        """Setup logging system"""
        log_level = getattr(logging, MONITORING_CONFIG['log_level'], logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('emergency_monitor.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def record_audio_sample(self, duration=3):
        """Record a short audio sample for analysis"""
        try:
            sample_rate = 16000
            audio_data = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()
            return audio_data.flatten(), sample_rate
            
        except Exception as e:
            self.logger.error(f"Audio recording failed: {e}")
            return None, None
    
    def analyze_emergency_conditions(self, stress_data, transcript, keywords_data):
        """Analyze all conditions to determine if emergency response is needed"""
        
        emergency_conditions = {
            'stress_critical': stress_data['smoothed_stress'] >= MONITORING_CONFIG['stress_threshold_critical'],
            'stress_high': stress_data['smoothed_stress'] >= MONITORING_CONFIG['stress_threshold_high'],
            'keywords_critical': keywords_data['severity'] == 'critical',
            'keywords_high': keywords_data['severity'] == 'high',
            'combined_threat': False
        }
        
        # Check for combined threats
        if (emergency_conditions['stress_high'] and 
            keywords_data['severity'] in ['high', 'critical']):
            emergency_conditions['combined_threat'] = True
        
        # Determine emergency type and priority
        if emergency_conditions['combined_threat']:
            return {
                'trigger_emergency': True,
                'source': 'CRITICAL: Stress + Keywords',
                'priority': 'CRITICAL',
                'confidence': max(stress_data['smoothed_stress'], keywords_data['confidence']),
                'details': f"Stress: {stress_data['smoothed_stress']:.2f}, Keywords: {keywords_data['severity']}"
            }
        
        elif emergency_conditions['stress_critical']:
            return {
                'trigger_emergency': True,
                'source': 'CRITICAL: Extreme Stress Level',
                'priority': 'HIGH',
                'confidence': stress_data['smoothed_stress'],
                'details': f"Stress level: {stress_data['smoothed_stress']:.2f}"
            }
        
        elif emergency_conditions['keywords_critical']:
            return {
                'trigger_emergency': True,
                'source': 'CRITICAL: Emergency Keywords',
                'priority': 'HIGH',
                'confidence': keywords_data['confidence'],
                'details': f"Keywords: {', '.join(keywords_data['detected_keywords'])}"
            }
        
        elif (emergency_conditions['stress_high'] and 
              stress_data['trend'] == 'increasing'):
            return {
                'trigger_emergency': True,
                'source': 'WARNING: Rising Stress Pattern',
                'priority': 'MEDIUM',
                'confidence': stress_data['smoothed_stress'],
                'details': f"Stress: {stress_data['smoothed_stress']:.2f}, Trend: {stress_data['trend']}"
            }
        
        else:
            return {
                'trigger_emergency': False,
                'source': 'Normal',
                'priority': 'NONE',
                'confidence': 0.0,
                'details': 'No emergency conditions detected'
            }
    
    def check_emergency_cooldown(self):
        """Check if enough time has passed since last emergency"""
        current_time = time.time()
        if current_time - self.last_emergency_time < MONITORING_CONFIG['emergency_cooldown']:
            remaining = MONITORING_CONFIG['emergency_cooldown'] - (current_time - self.last_emergency_time)
            return False, remaining
        return True, 0
    
    def process_monitoring_cycle(self):
        """Process one complete monitoring cycle"""
        try:
            # Record audio sample
            audio_data, sample_rate = self.record_audio_sample(
                duration=MONITORING_CONFIG['audio_check_interval']
            )
            
            if audio_data is None:
                return
            
            # Analyze stress
            stress_data = self.stress_analyzer.analyze_audio_stress(audio_data)
            
            # Transcribe audio
            transcript = self.transcriber.transcribe_audio(audio_data, sample_rate)
            
            # Analyze keywords
            keywords_data = self.keyword_analyzer.analyze_keywords(transcript)
            
            # Log current status
            self.logger.info(
                f"Monitor - Stress: {stress_data['smoothed_stress']:.2f}, "
                f"Keywords: {keywords_data['severity']}, "
                f"Transcript: '{transcript[:50]}...'"
            )
            
            # Check emergency conditions
            emergency_assessment = self.analyze_emergency_conditions(
                stress_data, transcript, keywords_data
            )
            
            if emergency_assessment['trigger_emergency']:
                # Check cooldown period
                can_trigger, cooldown_remaining = self.check_emergency_cooldown()
                
                if can_trigger:
                    self.logger.critical(f"🚨 EMERGENCY DETECTED: {emergency_assessment['source']}")
                    
                    # Trigger emergency response
                    emergency_result = check_and_trigger_emergency(
                        source=emergency_assessment['source'],
                        confidence=emergency_assessment['confidence'],
                        transcript=transcript,
                        keywords=keywords_data['detected_keywords']
                    )
                    
                    if emergency_result:
                        self.last_emergency_time = time.time()
                        self.logger.info(f"✅ Emergency response triggered: {emergency_result['emergency_id']}")
                    
                else:
                    self.logger.warning(
                        f"⏳ Emergency detected but in cooldown period "
                        f"({cooldown_remaining:.1f}s remaining)"
                    )
            
            # Display current status
            self.display_status(stress_data, keywords_data, transcript, emergency_assessment)
            
        except Exception as e:
            self.logger.error(f"Monitoring cycle error: {e}")
            import traceback
            traceback.print_exc()
    
    def display_status(self, stress_data, keywords_data, transcript, emergency_assessment):
        """Display current monitoring status"""
        
        # Clear screen (optional)
        # os.system('cls' if os.name == 'nt' else 'clear')
        
        print(f"\n{'='*80}")
        print(f"🛡️ SHIELDHER EMERGENCY MONITOR - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}")
        
        # Stress Status
        stress_level = stress_data['smoothed_stress']
        if stress_level >= MONITORING_CONFIG['stress_threshold_critical']:
            stress_indicator = "🔴 CRITICAL"
        elif stress_level >= MONITORING_CONFIG['stress_threshold_high']:
            stress_indicator = "🟡 HIGH"
        else:
            stress_indicator = "🟢 NORMAL"
        
        print(f"📊 STRESS LEVEL: {stress_indicator} ({stress_level:.2f})")
        print(f"📈 TREND: {stress_data['trend'].upper()}")
        
        # Keywords Status
        if keywords_data['severity'] == 'critical':
            keyword_indicator = "🔴 CRITICAL"
        elif keywords_data['severity'] == 'high':
            keyword_indicator = "🟡 HIGH"
        elif keywords_data['severity'] == 'medium':
            keyword_indicator = "🟠 MEDIUM"
        else:
            keyword_indicator = "🟢 NONE"
        
        print(f"🔍 KEYWORDS: {keyword_indicator}")
        if keywords_data['detected_keywords']:
            print(f"   Detected: {', '.join(keywords_data['detected_keywords'])}")
        
        # Current transcript
        if transcript:
            print(f"🎙️ TRANSCRIPT: \"{transcript[:60]}{'...' if len(transcript) > 60 else ''}\"")
        
        # Emergency status
        if emergency_assessment['trigger_emergency']:
            print(f"🚨 EMERGENCY STATUS: {emergency_assessment['priority']} - {emergency_assessment['source']}")
        else:
            print(f"✅ STATUS: MONITORING - {emergency_assessment['details']}")
        
        print(f"{'='*80}")
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        if self.monitoring_active:
            print("⚠️ Monitoring already active")
            return
        
        self.monitoring_active = True
        print("🔄 Starting emergency monitoring...")
        print("Press Ctrl+C to stop monitoring")
        
        try:
            while self.monitoring_active:
                self.process_monitoring_cycle()
                
                if MONITORING_CONFIG['continuous_monitoring']:
                    time.sleep(1)  # Short pause between cycles
                else:
                    break  # Single cycle mode
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            self.logger.error(f"Monitoring error: {e}")
        finally:
            self.monitoring_active = False
            print("✅ Emergency monitoring stopped")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
    
    def run_test_cycle(self):
        """Run a single test monitoring cycle"""
        print("🧪 Running test monitoring cycle...")
        MONITORING_CONFIG['continuous_monitoring'] = False
        self.process_monitoring_cycle()
        print("✅ Test cycle completed")

def main():
    """Main function"""
    print("🛡️ ShieldHer Emergency Detection System")
    print("="*50)
    
    # Initialize monitor
    monitor = EmergencyMonitor()
    
    # Configuration display
    print(f"📋 Configuration:")
    print(f"   Stress Threshold (High): {MONITORING_CONFIG['stress_threshold_high']}")
    print(f"   Stress Threshold (Critical): {MONITORING_CONFIG['stress_threshold_critical']}")
    print(f"   Audio Check Interval: {MONITORING_CONFIG['audio_check_interval']}s")
    print(f"   Emergency Cooldown: {MONITORING_CONFIG['emergency_cooldown']}s")
    
    # User menu
    while True:
        print("\n" + "="*50)
        print("OPTIONS:")
        print("1. Start Continuous Monitoring")
        print("2. Run Single Test Cycle")
        print("3. View Configuration")
        print("4. Exit")
        
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == '1':
            monitor.start_monitoring()
        elif choice == '2':
            monitor.run_test_cycle()
        elif choice == '3':
            print("\n📋 Current Configuration:")
            for key, value in MONITORING_CONFIG.items():
                print(f"   {key}: {value}")
        elif choice == '4':
            print("👋 Goodbye! Stay safe.")
            break
        else:
            print("❌ Invalid option. Please select 1-4.")

if __name__ == "__main__":
    main()