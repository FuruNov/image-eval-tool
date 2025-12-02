import json
import os
import streamlit as st

STATE_FILE = ".app_state.json"

class StateManager:
    def __init__(self, enabled=False):
        self.enabled = enabled
        self.state = {}
        if self.enabled:
            self.load_state()

    def load_state(self):
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE, "r") as f:
                    self.state = json.load(f)
            except Exception as e:
                print(f"Error loading state: {e}")
                self.state = {}

    def save_state(self, key):
        if not self.enabled:
            return
        
        # Get value from session_state
        # Get value from session_state
        if key in st.session_state:
            self.state[key] = st.session_state[key]
            self._save_to_disk()

    def set_value(self, key, value):
        if not self.enabled:
            return
        self.state[key] = value
        self._save_to_disk()

    def _save_to_disk(self):
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"Error saving state: {e}")

    def get_value(self, key, default):
        if self.enabled and key in self.state:
            return self.state[key]
        return default

# Singleton instance will be created in app.py
