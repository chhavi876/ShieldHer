import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf
import queue
import vosk
import json
import time
import scipy.signal
import threading
from datetime import datetime, timezone
import os
from pynput import keyboard

# === CONFIG ===
MODEL_PATH = r"C:\Users\MUSKAN\Desktop\ShieldHer\ai_model\voice_stress_model.tflite"
VOSK_MODEL_PATH = "vosk_model"
KEYWORDS = ["help", "emergency", "save me", "please help", "leave me", "stop", "no please", "help me", "help me","donot touch me",
    "i said no", "let me go", "get away from me", "you are scaring me", "i need help", "why are you doing this", "please stop", "call the police", "i donot feel safe",
    "this is not okay", "back off","stop following me","i am not okay",
    "you are hurting me", "donot come closer", "stay away", "this is harassment", "i want to leave"]
STRESS_THRESHOLD = 0.85
DURATION = 4
SAMPLE_RATE = 16000
BLOCKSIZE = 4000
MIN_AUDIO_LEVEL = 0.001  # Lowered threshold
KEYBOARD_INACTIVITY_THRESHOLD = 0  # No keypresses

def list_audio_devices():
    """List all available audio devices"""
    print("🎤 Available Audio Devices:")
    print("=" * 50)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        device_type = []
        if device['max_input_channels'] > 0:
            device_type.append("INPUT")
        if device['max_output_channels'] > 0:
            device_type.append("OUTPUT")
        
        print(f"{i}: {device['name']}")
        print(f"   Type: {' & '.join(device_type)}")
        print(f"   Channels: In={device['max_input_channels']}, Out={device['max_output_channels']}")
        print(f"   Sample Rate: {device['default_samplerate']}")
        print()
    
    # Show default devices
    try:
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        print(f"🎯 Default Input Device: {default_input['name']}")
        print(f"🎯 Default Output Device: {default_output['name']}")
        print()
    except:
        print("⚠️ Could not determine default devices")

def test_microphone():
    """Test microphone input with real-time monitoring"""
    print("🎤 Testing Microphone Input...")
    print("Speak into your microphone for 5 seconds...")
    print("You should see audio level indicators below:")
    print()
    
    audio_data = []
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Status: {status}")
        
        # Calculate RMS (volume level)
        rms = np.sqrt(np.mean(indata**2))
        
        # Visual audio level indicator
        level = int(rms * 1000)  # Scale for display
        bar = "█" * min(level, 50)
        print(f"\rAudio Level: {rms:.6f} |{bar:<50}| ", end="", flush=True)
        
        audio_data.append(indata.copy())
    
    try:
        with sd.InputStream(callback=callback, 
                          channels=1, 
                          samplerate=SAMPLE_RATE,
                          dtype=np.float32):
            time.sleep(5)
        
        print(f"\n\n📊 Test Results:")
        if audio_data:
            combined = np.concatenate(audio_data)
            max_level = np.max(np.abs(combined))
            rms_level = np.sqrt(np.mean(combined**2))
            print(f"   Max Audio Level: {max_level:.6f}")
            print(f"   RMS Audio Level: {rms_level:.6f}")
            print(f"   Total Samples: {len(combined)}")
            
            if max_level > 0.001:
                print("✅ Microphone is working!")
                return True, rms_level
            else:
                print("❌ No audio detected - microphone may be muted or not working")
                return False, 0
        else:
            print("❌ No audio data received")
            return False, 0
            
    except Exception as e:
        print(f"❌ Microphone test failed: {e}")
        return False, 0

class ShieldHerMonitor:
    def __init__(self, device_id=None, sensitivity_multiplier=1.0):
        self.audio_queue = queue.Queue()
        self.device_id = device_id
        self.sensitivity_multiplier = sensitivity_multiplier
        self.min_audio_level = MIN_AUDIO_LEVEL / sensitivity_multiplier
        self.key_count = 0
        self.last_key_check = time.time()
        self.interpreter = None
        self.vosk_model = None
        self.models_loaded = False
        self.is_monitoring = False
        self.setup_models()

    def setup_models(self):
        """Initialize ML models and other setup tasks"""
        print("🔧 Setting up models...")
        
        # Setup TensorFlow Lite model
        try:
            print(f"📁 Checking TFLite model at: {MODEL_PATH}")
            if not os.path.exists(MODEL_PATH):
                print(f"❌ TFLite model file not found at: {MODEL_PATH}")
                print("Please ensure the model file exists and the path is correct.")
            else:
                print("✅ TFLite model file found")
                self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
                print("✅ TFLite model loaded successfully")
                print(f"Input shape: {self.input_details[0]['shape']}")
                print(f"Output shape: {self.output_details[0]['shape']}")
        except Exception as e:
            print(f"❌ TFLite model loading error: {e}")
            self.interpreter = None

        # Setup Vosk model
        try:
            print(f"📁 Checking Vosk model at: {VOSK_MODEL_PATH}")
            print(f"🔍 Current Working Directory: {os.getcwd()}")
            
            if not os.path.exists(VOSK_MODEL_PATH):
                print(f"❌ Vosk model directory not found at: {VOSK_MODEL_PATH}")
                print("Please download a Vosk model and extract it to the correct path.")
                print("Visit: https://alphacephei.com/vosk/models")
            else:
                print("✅ Vosk model directory found")
                print(f"📁 Contents: {os.listdir(VOSK_MODEL_PATH)}")
                
                # Check for essential files
                required_files = ['am', 'graph', 'ivector']
                missing_files = [f for f in required_files if not os.path.exists(os.path.join(VOSK_MODEL_PATH, f))]
                
                if missing_files:
                    print(f"❌ Missing required files in Vosk model: {missing_files}")
                else:
                    print("✅ All required Vosk files present")
                    self.vosk_model = vosk.Model(VOSK_MODEL_PATH)
                    print("✅ Vosk model loaded successfully")
                    
        except Exception as e:
            print(f"❌ Vosk model loading error: {e}")
            print("This might be due to:")
            print("1. Incomplete model download")
            print("2. Corrupted model files")
            print("3. Incompatible model version")
            self.vosk_model = None

        # Check overall status
        if self.interpreter is not None and self.vosk_model is not None:
            self.models_loaded = True
            print("🎉 All models loaded successfully!")
        else:
            print("⚠️  Some models failed to load. Limited functionality available.")
            if self.interpreter is None:
                print("   - Stress detection will not work")
            if self.vosk_model is None:
                print("   - Keyword detection will not work")

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        
        rms = np.sqrt(np.mean(indata**2))
        max_val = np.max(np.abs(indata))
        self.audio_queue.put({
            'data': indata.copy(),
            'rms': rms,
            'max': max_val,
            'timestamp': time.time()
        })

    def detect_stress(self, audio_data):
        """Detect stress in audio using TensorFlow Lite model"""
        if self.interpreter is None:
            return 0.0
        
        try:
            # Extract features (you'll need to implement this based on your model)
            # This is a placeholder - replace with your actual feature extraction
            features = self.extract_features(audio_data)
            
            # Set input tensor
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            
            return float(output_data[0])
        except Exception as e:
            print(f"Stress detection error: {e}")
            return 0.0

    def extract_features(self, audio_data):
        """Extract features from audio data for stress detection"""
        # This is a placeholder - implement based on your model's requirements
        # Common features for stress detection include MFCC, spectral features, etc.
        try:
            # Example: extract MFCC features
            mfccs = librosa.feature.mfcc(y=audio_data, sr=SAMPLE_RATE, n_mfcc=13)
            features = np.mean(mfccs.T, axis=0)
            return features.reshape(1, -1).astype(np.float32)
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.zeros((1, 13), dtype=np.float32)

    def detect_keywords(self, audio_chunks_data):
        """Keyword detection with better audio handling"""
        if self.vosk_model is None:
            return "", []
            
        try:
            recognizer = vosk.KaldiRecognizer(self.vosk_model, SAMPLE_RATE)
            transcript = ""
            
            for chunk_info in audio_chunks_data:
                chunk = chunk_info['data'].flatten()
                
                # Skip very quiet chunks
                if chunk_info['rms'] < self.min_audio_level:
                    continue
                
                # Convert to int16 for Vosk
                audio_int16 = (chunk * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                if recognizer.AcceptWaveform(audio_bytes):
                    result = json.loads(recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        transcript += text + " "
            
            # Get final result
            final_result = json.loads(recognizer.FinalResult())
            final_text = final_result.get("text", "").strip()
            if final_text:
                transcript += final_text
            
            transcript = transcript.strip().lower()
            
            # Check for keywords
            detected_keywords = [kw for kw in KEYWORDS if kw.lower() in transcript]
            
            return transcript, detected_keywords
            
        except Exception as e:
            print(f"❌ Keyword detection error: {e}")
            return "", []

    def is_silence(self, audio_data):
        """Enhanced silence detection"""
        rms = np.sqrt(np.mean(audio_data**2))
        max_val = np.max(np.abs(audio_data))
        
        is_silent = rms < self.min_audio_level and max_val < (self.min_audio_level * 3)
        return is_silent, rms, max_val

    def on_key_press(self, key):
        self.key_count += 1

    def start_keyboard_listener(self):
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.start()
        print("⌨️  Keyboard listener started")

    def start_monitoring(self):
        """Start monitoring with enhanced debugging"""
        self.is_monitoring = True
        
        print("🟢 ShieldHer monitoring started")
        print(f"🎯 Audio sensitivity: {1/self.min_audio_level:.0f}x")
        print(f"🔊 Minimum audio level: {self.min_audio_level:.6f}")
        if self.device_id is not None:
            print(f"🎤 Using device ID: {self.device_id}")
        print("Press Ctrl+C to stop monitoring\n")
        
        self.start_keyboard_listener()
        
        try:
            # Configure audio stream
            stream_params = {
                'samplerate': SAMPLE_RATE,
                'blocksize': BLOCKSIZE,
                'dtype': np.float32,
                'channels': 1,
                'callback': self.audio_callback
            }
            
            if self.device_id is not None:
                stream_params['device'] = self.device_id
            
            with sd.InputStream(**stream_params):
                audio_buffer = []
                buffer_start_time = time.time()
                last_stats_time = time.time()
                
                while self.is_monitoring:
                    current_time = time.time()
                    
                    # Show periodic stats
                    if current_time - last_stats_time > 10:  # Every 10 seconds
                        queue_size = self.audio_queue.qsize()
                        buffer_size = len(audio_buffer)
                        print(f"📊 Queue: {queue_size}, Buffer: {buffer_size} chunks")
                        last_stats_time = current_time
                    
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
                            print(f"\n🎙️ Processing {len(audio_buffer)} audio chunks (Duration: {DURATION}s)...")
                            
                            # Calculate overall audio statistics
                            rms_values = [chunk['rms'] for chunk in audio_buffer]
                            max_values = [chunk['max'] for chunk in audio_buffer]
                            overall_rms = np.mean(rms_values)
                            overall_max = np.max(max_values)
                            
                            print(f"📊 Audio Stats - RMS: {overall_rms:.6f}, Max: {overall_max:.6f}")
                            
                            # Combine audio data
                            combined_audio = np.concatenate([chunk['data'].flatten() for chunk in audio_buffer])
                            
                            # Enhanced silence detection
                            is_silent, final_rms, final_max = self.is_silence(combined_audio)
                            
                            if not is_silent:
                                print(f"🔊 Audio detected! Processing...")
                                
                                # Stress detection
                                if self.interpreter is not None:
                                    stress_score = self.detect_stress(combined_audio)
                                    print(f"🧠 Stress Score: {stress_score:.4f} (Threshold: {STRESS_THRESHOLD})")
                                else:
                                    stress_score = 0.0
                                    print("⚠️ Stress detection unavailable (model not loaded)")
                                
                                # Keyword detection
                                if self.vosk_model is not None:
                                    transcript, detected_keywords = self.detect_keywords(audio_buffer)
                                    
                                    if transcript:
                                        print(f"🗣️ Transcript: \"{transcript}\"")
                                    else:
                                        print("🤔 No clear speech detected")
                                    
                                    if detected_keywords:
                                        print(f"⚠️ Emergency keywords: {detected_keywords}")
                                else:
                                    transcript, detected_keywords = "", []
                                    print("⚠️ Keyword detection unavailable (model not loaded)")
                                
                                # Alert logic (only if models are loaded)
                                if self.interpreter is not None and self.vosk_model is not None:
                                    stress_detected = stress_score >= STRESS_THRESHOLD
                                    keywords_detected = len(detected_keywords) > 0
                                    
                                    if stress_detected or keywords_detected:
                                        print("🚨 EMERGENCY CONDITIONS MET!")
                                        
                                        # Try to import and use emergency dispatcher
                                        try:
                                            from emergency_dispatcher import build_alert, send_alert
                                            alert_payload = build_alert(transcript, detected_keywords, stress_score)
                                            send_alert(alert_payload)
                                        except ImportError:
                                            print("⚠️ Emergency dispatcher module not found - alert not sent")
                                        except Exception as e:
                                            print(f"❌ Failed to send alert: {e}")
                                    else:
                                        print("✅ No emergency detected")
                                else:
                                    print("ℹ️ Running in test mode - no alerts triggered")
                            else:
                                print(f"🔇 Silent period (RMS: {final_rms:.6f}, Max: {final_max:.6f})")
                        
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
                    
        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped by user")
        except Exception as e:
            print(f"❌ Error during monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_monitoring = False

def main():
    """Main function with device selection and testing"""
    print("🛡️ ShieldHer Audio Diagnostic Tool")
    print("=" * 50)
    
    # Step 1: List audio devices
    list_audio_devices()
    
    # Step 2: Test microphone
    print("Step 1: Testing default microphone...")
    mic_working, audio_level = test_microphone()
    
    if not mic_working:
        print("\n❌ Microphone test failed!")
        print("Troubleshooting steps:")
        print("1. Check if microphone is connected and not muted")
        print("2. Check Windows sound settings")
        print("3. Try selecting a different device below")
        print("4. Grant microphone permissions to Python/your IDE")
        
        choice = input("\nTry different device? (y/n): ").lower()
        if choice == 'y':
            try:
                device_id = int(input("Enter device ID: "))
                sd.default.device[0] = device_id  # Set input device
                print(f"Testing device {device_id}...")
                mic_working, audio_level = test_microphone()
            except:
                print("Invalid device ID")
    
    if mic_working:
        print(f"\n✅ Microphone working! Audio level: {audio_level:.6f}")
        
        # Determine sensitivity multiplier based on audio level
        if audio_level < 0.001:
            sensitivity = 10.0
            print("🔊 Low audio detected - using high sensitivity")
        elif audio_level < 0.01:
            sensitivity = 3.0
            print("🔊 Medium audio detected - using medium sensitivity")
        else:
            sensitivity = 1.0
            print("🔊 Good audio detected - using normal sensitivity")
        
        input("\nPress Enter to start monitoring...")
        
        # Start monitoring
        device_id = getattr(sd.default.device[0], '__int__', lambda: None)()
        monitor = ShieldHerMonitor(device_id=device_id, sensitivity_multiplier=sensitivity)
        monitor.start_monitoring()
    else:
        print("❌ Cannot start monitoring - microphone not working")

if __name__ == "__main__":
    main()