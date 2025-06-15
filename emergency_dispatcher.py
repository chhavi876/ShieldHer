import json
import os
import time
import requests
import soundfile as sf
from datetime import datetime, timezone
from twilio.rest import Client

# Twilio credentials
TWILIO_SID = "ACcb81d49fd97b61fe1cdd1c4f0c152356"
TWILIO_AUTH = "39a93bc7b68b864aa5139ab05f50c3b6"
TWILIO_SMS_NUMBER = "+19064482160"

CONTACTS_FILE = "contacts.json"
LOG_DIR = "alert_logs"
RECORDINGS_DIR = "recordings"

client = Client(TWILIO_SID, TWILIO_AUTH)

def get_real_location():
    try:
        res = requests.get("https://ipinfo.io/json")
        if res.status_code == 200:
            data = res.json()
            lat, lon = map(float, data["loc"].split(","))
            return {"lat": lat, "lon": lon}
    except:
        pass
    return {"lat": 28.6139, "lon": 77.209}  # fallback

def load_contacts():
    try:
        with open(CONTACTS_FILE, "r") as f:
            data = json.load(f)
        return data.get("emergency_contacts", [])
    except Exception as e:
        print(f"❌ Failed to load contacts: {e}")
        return []

def build_alert(transcript, keywords, stress_score, audio_data=None, sample_rate=16000):
    # Save recording
    os.makedirs(RECORDINGS_DIR, exist_ok=True)
    filename = f"alert_{int(time.time())}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)

    if audio_data is not None:
        try:
            sf.write(filepath, audio_data, sample_rate)
            print(f"🔊 Audio recorded: {filepath}")
        except Exception as e:
            print(f"❌ Failed to save audio: {e}")
            filepath = None
    else:
        filepath = None

    return {
        "status": "emergency",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": get_real_location(),
        "stress_score": round(stress_score, 4),
        "keywords": keywords,
        "transcript": transcript,
        "recording": filepath
    }

def format_message(alert):
    return (
        "🚨 ShieldHer Alert 🚨\n"
        f"Time: {alert['timestamp']}\n"
        f"Location: {alert['location']['lat']}, {alert['location']['lon']}\n"
        f"Stress Score: {alert['stress_score']}\n"
        f"Keywords: {', '.join(alert['keywords']) or 'None'}\n"
        f"Transcript: {alert['transcript'] or 'No speech detected'}"
    )

def send_alert(alert):
    contacts = load_contacts()
    if not contacts:
        print("⚠️ No contacts found.")
        return

    message_text = format_message(alert)

    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for contact in contacts:
        name = contact.get("name")
        phone = contact.get("phone")

        print(f"\n📤 Sending alert to {name} ({phone})...")
        log_file = os.path.join(LOG_DIR, f"alert_{name}_{timestamp}.json")
        with open(log_file, "w") as f:
            json.dump(alert, f, indent=2)
        print(f"✅ Alert logged to {log_file}")

        if phone:
            try:
                sms = client.messages.create(
                    to=phone,
                    from_=TWILIO_SMS_NUMBER,
                    body=message_text
                )
                print(f"📩 SMS sent — SID: {sms.sid}")
            except Exception as e:
                print(f"❌ SMS failed for {name}: {e}")
        else:
            print(f"⚠️ No phone number for {name}")
