import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="UWB Live Lab", layout="wide")

st.title("📡 Ultra-Wideband (UWB) Live Lab")
st.write("Interactive simulation of UWB signals without external datasets or models.")

# Sidebar controls
pulse_type = st.sidebar.selectbox("Select Pulse Type", ["Gaussian", "Square", "Multi-Carrier"])
num_pulses = st.sidebar.slider("Number of Pulses", 1, 10, 3)
pulse_width = st.sidebar.slider("Pulse Width (ns)", 0.1, 5.0, 0.5, 0.1)
carrier_freq = st.sidebar.slider("Carrier Frequency (GHz)", 1.0, 10.0, 3.0, 0.1)
noise_level = st.sidebar.slider("Noise Level", 0.0, 0.5, 0.05, 0.01)

# Time vector
Fs = 100e9  # Sampling rate 100 GHz
T = 20e-9   # 20 ns window
t = np.linspace(-T/2, T/2, int(Fs*T))

# Generate pulses
if pulse_type == "Gaussian":
    pulse = np.exp(-0.5 * (t / (pulse_width * 1e-9))**2)
elif pulse_type == "Square":
    pulse = np.where(np.abs(t) < (pulse_width * 1e-9) / 2, 1.0, 0.0)
elif pulse_type == "Multi-Carrier":
    pulse = np.sin(2*np.pi*carrier_freq*1e9*t) * np.exp(-0.5 * (t / (pulse_width * 1e-9))**2)

# Repetition
uwb_signal = np.tile(pulse, num_pulses)

# Add noise
uwb_signal += noise_level * np.random.randn(len(uwb_signal))

# Plot Time Domain
fig1, ax1 = plt.subplots()
ax1.plot(t[:2000]*1e9, uwb_signal[:2000])  # Show first 2000 samples
ax1.set_title("Time-Domain UWB Signal")
ax1.set_xlabel("Time (ns)")
ax1.set_ylabel("Amplitude")
st.pyplot(fig1)

# Frequency Domain
fft_spectrum = np.abs(np.fft.fft(uwb_signal))**2
freqs = np.fft.fftfreq(len(uwb_signal), 1/Fs)

fig2, ax2 = plt.subplots()
ax2.plot(freqs[:len(freqs)//2]*1e-9, fft_spectrum[:len(freqs)//2])
ax2.set_title("Frequency Spectrum of UWB Signal")
ax2.set_xlabel("Frequency (GHz)")
ax2.set_ylabel("Power")
st.pyplot(fig2)

st.success("✅ UWB Signal Generated Successfully!")
