import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.signal import butter, lfilter, freqz, resample_poly
from scipy.signal.windows import hann

# -------------------------
# Helpers & Core DSP Blocks
# -------------------------
def db10(x, eps=1e-12):
    return 10*np.log10(np.maximum(x, eps))

def butter_bandpass(low, high, fs, order=5):
    nyq = 0.5*fs
    lowc = low/nyq
    highc = high/nyq
    return butter(order, [lowc, highc], btype='band')

def butter_bandstop(low, high, fs, order=5):
    nyq = 0.5*fs
    lowc = low/nyq
    highc = high/nyq
    return butter(order, [lowc, highc], btype='bandstop')

def filt(b, a, x):
    return lfilter(b, a, x)

def psd_welch(x, fs, nfft):
    # Simple Welch-like PSD for speed (no overlap)
    x = x - np.mean(x)
    seglen = nfft
    nseg = len(x)//seglen
    if nseg < 1:
        nseg = 1
        seglen = len(x)
    x = x[:nseg*seglen].reshape(nseg, seglen)
    win = hann(seglen, sym=False)
    U = (win**2).sum()
    X = np.fft.rfft(x*win, n=seglen, axis=1)
    S = (np.abs(X)**2)/(U*fs)
    Pxx = S.mean(axis=0)
    f = np.fft.rfftfreq(seglen, 1/fs)
    return f, Pxx

def gaussian_monocycle(t, tau):
    # First derivative of Gaussian (monocycle) with unit energy scaling
    g = np.exp(-t**2/(2*tau**2))
    dg = -(t/(tau**2))*g
    # Normalize energy
    e = np.sum(dg**2)
    if e > 0:
        dg = dg/np.sqrt(e)
    return dg

def time_hopping_indices(Nsym, Thop, fs):
    max_idx = int(Thop*fs)
    hops = np.random.randint(0, max_idx, size=Nsym)
    return hops

def add_awgn(x, snr_db):
    # SNR per sample; normalize to unit power then scale noise
    p = np.mean(np.abs(x)**2)
    if p == 0:
        return x
    x_n = x/np.sqrt(p)
    snr_lin = 10**(snr_db/10)
    n = (np.random.randn(*x.shape)+1j*np.random.randn(*x.shape))/np.sqrt(2*snr_lin)
    y = x_n + n
    return y*np.sqrt(p)

def add_nb_interferer(fs, N, f0, snr_db, kind="tone"):
    t = np.arange(N)/fs
    if kind == "tone":
        s = np.exp(1j*2*np.pi*f0*t)
    elif kind == "qpsk":
        # simple QPSK narrowband at f0
        sym_rate = f0/10 if f0/10 > 1000 else 1000
        M = int(np.ceil(N*sym_rate/fs))
        bits = np.random.randint(0, 2, size=2*M)
        sym = (2*bits[0::2]-1)+1j*(2*bits[1::2]-1)
        # upsample and mix
        up = int(np.floor(fs/sym_rate))
        base = np.repeat(sym, up)[:N]
        s = base*np.exp(1j*2*np.pi*f0*t)
    else:
        s = np.zeros(N, dtype=complex)
    # scale to target SNR vs unit-power signal
    s = s/np.sqrt(np.mean(np.abs(s)**2)+1e-12)
    s = add_awgn(s, snr_db)  # reuse AWGN to scale
    return s

def multipath_channel(x, fs, taps_delays_ns, taps_gains_db):
    y = np.zeros_like(x, dtype=complex)
    for d_ns, g_db in zip(taps_delays_ns, taps_gains_db):
        delay = int(np.round(d_ns*1e-9*fs))
        gain = 10**(g_db/20)
        if delay < len(x):
            y[delay:] += gain*x[:len(x)-delay]
    return y

def mrc_rake(y, template, fs, taps_delays_ns, taps_gains_db):
    # Correlate around each expected path and weight by path gain (MRC)
    N = len(y)
    symcorr = 0
    for d_ns, g_db in zip(taps_delays_ns, taps_gains_db):
        delay = int(np.round(d_ns*1e-9*fs))
        if delay+len(template) <= N:
            r = np.vdot(template, y[delay:delay+len(template)])
            w = 10**(g_db/20)
            symcorr += w*r
    return symcorr

def eye_diagram(x, sps, ntraces=50):
    traces = []
    for k in range(ntraces):
        start = k*sps
        if start+2*sps <= len(x):
            traces.append(np.real(x[start:start+2*sps]))
    return np.array(traces)

# -------------------------
# I-UWB Signal Generators
# -------------------------
def gen_iuwb_th_ppm(bits, fs, Rs, tau_ps, Thop_ns, jitter_ps=0.0):
    # one pulse per bit; PPM: bit 0 -> early, bit 1 -> late by delta
    Tb = 1/Rs
    Nsym = len(bits)
    N = int(np.ceil(Nsym*Tb*fs + 10*fs/Rs))
    x = np.zeros(N, dtype=complex)
    tau = tau_ps*1e-12
    Thop = Thop_ns*1e-9
    hops = time_hopping_indices(Nsym, Thop, fs)
    ppm_shift = int(0.15*fs/Rs)  # 15% of Tb for clear separation
    t_p = np.arange(-6*tau, 6*tau, 1/fs)
    p = gaussian_monocycle(t_p, tau)
    pL = len(p)
    for i, b in enumerate(bits):
        base = int(i*Tb*fs) + hops[i]
        pos = base + (ppm_shift if b else 0)
        if jitter_ps > 0:
            j = int(np.round(np.random.randn()*jitter_ps*1e-12*fs))
        else:
            j = 0
        idx = pos + j
        if idx >= 0 and idx+pL < N:
            x[idx:idx+pL] += p
    return x

def gen_iuwb_ds(bits, fs, Rs, tau_ps, chips_per_bit=8):
    Tb = 1/Rs
    Tc = Tb/chips_per_bit
    N = int(np.ceil(len(bits)*Tb*fs + 4*fs/Rs))
    x = np.zeros(N, dtype=complex)
    tau = tau_ps*1e-12
    t_p = np.arange(-6*tau, 6*tau, 1/fs)
    p = gaussian_monocycle(t_p, tau)
    pL = len(p)
    # simple ±1 PN derived from LFSR-like sequence
    np.random.seed(7)
    pn = 2*np.random.randint(0, 2, size=chips_per_bit)-1
    for i, b in enumerate(bits):
        chips = pn*(1 if b else -1)
        for c, val in enumerate(chips):
            t0 = int(np.round((i*Tb + c*Tc)*fs))
            if t0+pL < N:
                x[t0:t0+pL] += val*p
    return x

def gen_iuwb_tr(bits, fs, Rs, tau_ps, ref_spacing=0.35):
    # Transmitted-Reference: send a known ref pulse then data pulse with ± polarity
    Tb = 1/Rs
    N = int(np.ceil(len(bits)*Tb*fs + 6*fs/Rs))
    x = np.zeros(N, dtype=complex)
    tau = tau_ps*1e-12
    t_p = np.arange(-6*tau, 6*tau, 1/fs)
    p = gaussian_monocycle(t_p, tau)
    pL = len(p)
    ref_off = int(ref_spacing*Tb*fs)
    for i, b in enumerate(bits):
        base = int(i*Tb*fs)
        if base+pL < N and base+ref_off+pL < N:
            x[base:base+pL] += p              # reference pulse
            x[base+ref_off:base+ref_off+pL] += (1 if b else -1)*p  # data pulse
    return x

# -------------------------
# MC-UWB (OFDM-like) Blocks
# -------------------------
def gen_mc_ofdm(bits, fs, Nfft, subcarrier_spacing, active_bins, cp_frac=0.1, hop=False):
    # Baseband OFDM, BPSK mapping on selected bins; optional simple hop (bin rotation)
    Ns = int(len(bits)/len(active_bins))
    bits = bits[:Ns*len(active_bins)]
    symb = (2*bits-1).astype(float)  # BPSK
    symb = symb.reshape(Ns, len(active_bins))
    cp_len = int(cp_frac*Nfft)
    s = []
    for k in range(Ns):
        X = np.zeros(Nfft, dtype=complex)
        bins = active_bins.copy()
        if hop:
            bins = np.roll(bins, k % len(bins))
        X[bins] = symb[k] + 0j
        x = np.fft.ifft(X)
        xcp = np.concatenate([x[-cp_len:], x])
        s.append(xcp)
    bb = np.concatenate(s)
    Ts = 1/(subcarrier_spacing*Nfft)  # OFDM symbol (no CP)
    fs_ofdm = (Nfft+cp_len)/Ts/(Nfft)  # effective sampling from chosen spacing
    # resample to requested fs if needed
    if abs(fs_ofdm - fs) > 1:
        # rational resample
        up = int(np.ceil(fs))
        down = int(np.ceil(fs_ofdm))
        bb = resample_poly(bb, up, down)
        fs_eff = fs*up/down  # not used further; just align length
        return bb[:int(len(bb)*fs/fs_eff)]
    return bb

def notch_band(bb, fs, notch_low, notch_high):
    b, a = butter_bandstop(notch_low, notch_high, fs, order=6)
    return lfilter(b, a, bb)

# -------------------------
# Receivers
# -------------------------
def detect_threshold(x):
    # very simple leading edge/energy detector
    thr = 3*np.std(x.real)
    idx = np.where(np.abs(x.real) > thr)[0]
    return idx[0] if len(idx) else None

def corr_detect_symbol(y, template):
    r = np.correlate(y, template, mode='valid')
    return np.argmax(np.abs(r)), r

def tr_autocorr_detect(y, ref_off, win):
    # correlate y[n] with y[n + ref_off] in a small window
    N = len(y)
    out = []
    for n in range(0, N-ref_off-win, win):
        a = y[n:n+win]
        b = y[n+ref_off:n+ref_off+win]
        out.append(np.vdot(a, b))
    return np.array(out)

def ber_bpsk(a_hat, a_true):
    return np.mean((np.sign(a_hat.real) != np.sign(a_true.real)).astype(float))

# -------------------------
# Streamlit UI
# -------------------------
st.set_page_config(page_title="UWB Live Lab", layout="wide")
st.title("⚡ Ultra-Wideband (UWB) Live Lab — I-UWB & MC-UWB (no datasets, no models)")

with st.sidebar:
    st.header("Signal Family")
    mode = st.radio("Choose UWB Type", ["I-UWB (Impulse)", "MC-UWB (OFDM-UWB)"])

    st.markdown("### Channel & Interference")
    use_multipath = st.checkbox("Multipath Channel (Indoor CM-like)", value=True)
    rake_enable = st.checkbox("RAKE Combine (MRC)", value=True)
    nb_on_uwb = st.checkbox("Add Narrowband Interferer (NB→UWB)", value=True)
    uwb_on_nb = st.checkbox("NB Receiver Filter to Measure UWB→NB", value=False)

    st.markdown("### Noise / Eb/N0")
    ebn0 = st.slider("Eb/N0 (dB)", -5, 30, 10, 1)
    ber_sweep = st.checkbox("Quick BER Sweep", value=False)

    st.markdown("### Rendering")
    nfft = st.select_slider("PSD FFT size", options=[1024, 2048, 4096, 8192], value=4096)

# Common settings
fs = 200e6  # 200 MHz baseband sample rate (demo-friendly)
Nsym = 256
rng = np.random.default_rng(42)
bits = rng.integers(0, 2, size=Nsym)

# Multipath profile (loosely inspired by indoor UWB)
taps_delays_ns = [0, 10, 20, 35, 60] if use_multipath else [0]
taps_gains_db  = [0, -3, -6, -9, -12] if use_multipath else [0]

# Narrowband interferer
f0_nb = 20e6  # 20 MHz offset inside our 200 MHz baseband
nb_kind = "qpsk"

# -------------------------
# Generate & Process
# -------------------------
if mode == "I-UWB (Impulse)":
    colA, colB, colC = st.columns(3)
    with colA:
        st.subheader("I-UWB Parameters")
        scheme = st.selectbox("Scheme", ["TH-PPM", "DS-UWB", "TR-UWB"])
        Rs = st.slider("Bit Rate (kb/s)", 50, 1000, 250, 50)*1e3
        tau_ps = st.slider("Pulse Width τ (ps)", 50, 600, 200, 10)
        Thop_ns = st.slider("Time-Hopping Slot (ns)", 1, 300, 60, 1)
        jitter_ps = st.slider("Timing Jitter (ps)", 0, 120, 20, 5)

    # Generate signal
    if scheme == "TH-PPM":
        x = gen_iuwb_th_ppm(bits, fs, Rs, tau_ps, Thop_ns, jitter_ps)
        template = gen_iuwb_th_ppm(np.array([1]), fs, Rs, tau_ps, Thop_ns, 0)
    elif scheme == "DS-UWB":
        x = gen_iuwb_ds(bits, fs, Rs, tau_ps, chips_per_bit=8)
        template = gen_iuwb_ds(np.array([1]), fs, Rs, tau_ps, chips_per_bit=8)
    else:
        x = gen_iuwb_tr(bits, fs, Rs, tau_ps, ref_spacing=0.35)
        template = gen_iuwb_tr(np.array([1]), fs, Rs, tau_ps, ref_spacing=0.35)

    # Channel + Interference + Noise
    y = multipath_channel(x, fs, taps_delays_ns, taps_gains_db)
    if nb_on_uwb:
        nb = add_nb_interferer(fs, len(y), f0_nb, 20, nb_kind)
        y = y + 0.3*nb  # moderate interferer power
    y = add_awgn(y, ebn0)

    # RAKE & simple detection demo (symbol-level sketch)
    rake_out = None
    if rake_enable and use_multipath:
        # Use small template window
        tpl = template[:min(1024, len(template))]
        rake_out = mrc_rake(y.astype(complex), tpl, fs, taps_delays_ns, taps_gains_db)

    # NB Receiver (measuring UWB→NB interference)
    nb_out_power = None
    if uwb_on_nb:
        bnb, anb = butter_bandpass(f0_nb-2e6, f0_nb+2e6, fs, order=6)
        z_nb = filt(bnb, anb, y)
        nb_out_power = 10*np.log10(np.mean(np.abs(z_nb)**2)+1e-12)

    # Visuals
    with colB:
        st.subheader("Time Domain (excerpt)")
        Lshow = min(6000, len(y))
        t_ax = np.arange(Lshow)/fs*1e6
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_ax, y=np.real(y[:Lshow]), name="y(t) real"))
        fig.update_layout(xaxis_title="Time (μs)", yaxis_title="Amplitude", height=280)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Autocorrelation / TR Metric")
        ref_off = int(0.35*(1/Rs)*fs)
        ac = tr_autocorr_detect(y.real, ref_off, win=max(64, int(0.03*fs/Rs)))
        fig_ac = go.Figure()
        fig_ac.add_trace(go.Scatter(y=np.abs(ac), name="|TR auto|"))
        fig_ac.update_layout(xaxis_title="Window index", yaxis_title="Metric", height=260)
        st.plotly_chart(fig_ac, use_container_width=True)

    with colC:
        st.subheader("PSD vs. Stylized UWB Mask")
        f, Pxx = psd_welch(y.real, fs, nfft)
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=f/1e6, y=db10(Pxx), name="PSD (dB/Hz)"))
        # Stylized indoor UWB mask (normalized): −41.3 dBm/MHz ~ flat band; here a simple box
        mask = np.interp(f, [0, 3e6, 30e6, 90e6, 100e6], [-90, -90, -75, -75, -90])
        figp.add_trace(go.Scatter(x=f/1e6, y=mask, name="Stylized Mask", line=dict(dash="dash")))
        figp.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Power (dB)", height=340)
        st.plotly_chart(figp, use_container_width=True)

    # Eye diagram (symbol stability)
    st.subheader("Eye Diagram (I-UWB, baseband samples)")
    sps = max(8, int(fs/Rs))
    eye = eye_diagram(y.real, sps, ntraces=40)
    if eye.size > 0:
        fig_eye = go.Figure()
        for tr in eye[:30]:
            fig_eye.add_trace(go.Scatter(y=tr, mode="lines", showlegend=False, opacity=0.4))
        fig_eye.update_layout(xaxis_title="Samples", yaxis_title="Amplitude", height=260)
        st.plotly_chart(fig_eye, use_container_width=True)
    else:
        st.info("Increase symbol rate or sampling to view eye diagram.")

    # Quick BER sweep (coarse)
    if ber_sweep:
        st.subheader("BER vs. Eb/N0 (sketch)")
        EbN0s = np.arange(-2, 19, 2)
        BER = []
        tpl = template[:min(512, len(template))].real
        tpl = tpl/np.sqrt(np.sum(tpl**2)+1e-12)
        for snr in EbN0s:
            yy = multipath_channel(x, fs, taps_delays_ns, taps_gains_db)
            if nb_on_uwb:
                nb = add_nb_interferer(fs, len(yy), f0_nb, 20, nb_kind)
                yy = yy + 0.3*nb
            yy = add_awgn(yy, snr)
            # Symbol-wise correlation over a coarse grid
            step = max(1, int(len(yy)/len(bits)))
            det = []
            k = 0
            for b in bits:
                seg = yy[k:k+len(tpl)]
                if len(seg) < len(tpl): break
                c = np.vdot(tpl, seg.real)
                det.append(1 if c.real > 0 else 0)
                k += step
            det = np.array(det[:len(bits)])
            BER.append(np.mean(det != bits[:len(det)]))
        figb = go.Figure()
        figb.add_trace(go.Scatter(x=EbN0s, y=BER, mode="lines+markers", name="I-UWB"))
        figb.update_layout(xaxis_title="Eb/N0 (dB)", yaxis_title="BER", yaxis_type="log", height=320)
        st.plotly_chart(figb, use_container_width=True)

    if rake_out is not None:
        st.success(f"RAKE (MRC) combined metric magnitude: {np.abs(rake_out):.3f}")
    if nb_out_power is not None:
        st.warning(f"NB receiver output power (UWB→NB leakage): {nb_out_power:.1f} dB")

# -------------------------
# MC-UWB (OFDM) Branch
# -------------------------
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("MC-UWB Parameters")
        Nfft = st.selectbox("FFT Size", [64, 128, 256, 512], index=2)
        sc_spacing = st.slider("Subcarrier Spacing (kHz)", 50, 500, 125, 25)*1e3
        n_active = st.slider("Active Subcarriers", 8, Nfft//2, min(64, Nfft//2), 4)
        hop = st.checkbox("Enable Simple Frequency Hopping", value=True)
        notch = st.checkbox("Apply Spectral Notch (Protect NB)", value=True)

    # Choose active bins spread across band
    bins = np.linspace(1, Nfft//2-1, n_active, dtype=int)
    bb = gen_mc_ofdm(bits, fs, Nfft, sc_spacing, bins, cp_frac=0.1, hop=hop)

    # Spectral notch around NB interferer
    if notch:
        bb = notch_band(bb, fs, f0_nb-3e6, f0_nb+3e6)

    # Channel & interference
    y = multipath_channel(bb, fs, taps_delays_ns, taps_gains_db)
    if nb_on_uwb:
        nb = add_nb_interferer(fs, len(y), f0_nb, 18, "tone")
        y = y + 0.25*nb
    y = add_awgn(y, ebn0)

    # Simple OFDM receiver (no CFO/phase tracking)
    # Recreate framing from choices
    cp_len = int(0.1*Nfft)
    sym_len = Nfft+cp_len
    Ns = len(y)//sym_len
    y = y[:Ns*sym_len]
    y = y.reshape(Ns, sym_len)
    y = y[:, cp_len:]  # drop CP
    Y = np.fft.fft(y, axis=1)
    rx_bins = Y[:, bins]
    a_hat = np.sign(rx_bins.real).ravel()
    a_true = (2*bits[:len(a_hat)]-1)
    ber = ber_bpsk(a_hat, a_true)

    with col2:
        st.subheader("Constellation (BPSK bins)")
        figc = go.Figure()
        pts = rx_bins.ravel()
        figc.add_trace(go.Scatter(x=pts.real, y=pts.imag, mode="markers", name="Bins"))
        figc.update_layout(xaxis_title="I", yaxis_title="Q", height=280)
        st.plotly_chart(figc, use_container_width=True)

        st.subheader("PSD")
        f, Pxx = psd_welch(y.ravel().real, fs, nfft)
        figp = go.Figure()
        figp.add_trace(go.Scatter(x=f/1e6, y=db10(Pxx), name="PSD"))
        # Stylized UWB mask
        mask = np.interp(f, [0, 3e6, 30e6, 90e6, 100e6], [-90, -90, -75, -75, -90])
        figp.add_trace(go.Scatter(x=f/1e6, y=mask, name="Stylized Mask", line=dict(dash="dash")))
        figp.update_layout(xaxis_title="Frequency (MHz)", yaxis_title="Power (dB)", height=320)
        st.plotly_chart(figp, use_container_width=True)

    with col3:
        st.subheader("Time Domain (excerpt)")
        Lshow = min(6000, len(bb))
        t_ax = np.arange(Lshow)/fs*1e6
        figt = go.Figure()
        figt.add_trace(go.Scatter(x=t_ax, y=np.real(bb[:Lshow]), name="bb real"))
        figt.update_layout(xaxis_title="Time (μs)", yaxis_title="Amp", height=280)
        st.plotly_chart(figt, use_container_width=True)

        st.subheader("BER (This Run)")
        st.success(f"MC-UWB OFDM BPSK BER: {ber:.3e}")

    if ber_sweep:
        st.subheader("BER vs. Eb/N0 (MC-UWB)")
        EbN0s = np.arange(-2, 19, 2)
        BER = []
        for snr in EbN0s:
            yy = multipath_channel(bb, fs, taps_delays_ns, taps_gains_db)
            if nb_on_uwb:
                nb = add_nb_interferer(fs, len(yy), f0_nb, 18, "tone")
                yy = yy + 0.25*nb
            yy = add_awgn(yy, snr)
            Ns = len(yy)//sym_len
            yy = yy[:Ns*sym_len].reshape(Ns, sym_len)[:, cp_len:]
            YY = np.fft.fft(yy, axis=1)
            rxb = YY[:, bins]
            ah = np.sign(rxb.real).ravel()
            at = (2*bits[:len(ah)]-1)
            BER.append(ber_bpsk(ah, at))
        figb = go.Figure()
        figb.add_trace(go.Scatter(x=EbN0s, y=BER, mode="lines+markers", name="MC-UWB"))
        figb.update_layout(xaxis_title="Eb/N0 (dB)", yaxis_title="BER", yaxis_type="log", height=320)
        st.plotly_chart(figb, use_container_width=True)

# Footer note
st.caption("All signals are synthesized on the fly. No datasets or pretrained models are used. \
This app illustrates I-UWB & MC-UWB signals, interference/coexistence, and receiver concepts (TR, correlation, RAKE).")
