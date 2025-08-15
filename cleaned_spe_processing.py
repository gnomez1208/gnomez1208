!pip install uproot

from google.colab import files

print("Please select 'output03048.root' from your Ubuntu machine.")
uploaded = files.upload()
#saved onto google colab local file environment


import uproot
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.signal import butter, lfilter, filtfilt
import plotly.graph_objects as go

'''
-----------things to note from MIDAS-----------
y axis - ADC counts per 4ns
- most if not all in range of 7000-8000 (wide)
- tigher range could be 7200 - 7600
x axis - time in ns (window for each waveform is 5000 ns)

'''



def butter_highpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print(f"High-pass filter: cutoff={cutoff}, fs={fs}, nyq={nyq}, normal_cutoff={normal_cutoff}")
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return filtfilt(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    print(f"Low-pass filter: cutoff={cutoff}, fs={fs}, nyq={nyq}, normal_cutoff={normal_cutoff}")
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)


def moving_average_filter(data, window_size):
    """
    Applies a moving average filter to the data. Be careful with bounds!
    Ended up needing to add a filtfilt function so that the df values length were conserved
    """
    a = 1
    b = np.ones(window_size) / window_size

    return filtfilt(b, a, data)

file_path = "/content/output03048.root"

#  1. DEFINING EXPERIMENTAL PARAMETERS
# ==============================================================================
sampling_period_ns = 4
integration_window_ns = 250
pre_trigger_ns = 200
pre_trigger_samples = int(pre_trigger_ns / sampling_period_ns)
sampling_frequency = 1 / (sampling_period_ns * 1e-9) # In Hz

# --- Select and load a single waveform ---
waveform_index_to_plot = 709

with uproot.open(file_path) as file:
    key = file.keys()[waveform_index_to_plot]
    waveform_np = file[key].values()

# --- Create the initial DataFrame ---
time_ns = np.arange(len(waveform_np)) * sampling_period_ns
df = pd.DataFrame({'Time (ns)': time_ns, 'Raw ADC': waveform_np})

# --- Perform baseline subtraction ---
baseline = df['Raw ADC'][:pre_trigger_samples].mean()
df['Baseline Subtracted'] = df['Raw ADC'] - baseline

# --- Apply filters and add them as new columns to the DataFrame ---
print("Applying filters...")

# Apply a high-pass filter
df['High-Pass'] = butter_highpass_filter(df['Baseline Subtracted'], cutoff=1e6, fs=sampling_frequency)

# Apply a low-pass filter
df['Low-Pass'] = butter_lowpass_filter(df['Baseline Subtracted'], cutoff=10e6, fs=sampling_frequency)

# Create a band-pass filter by applying the low-pass to the high-passed signal
df['Band-Pass'] = butter_lowpass_filter(df['High-Pass'], cutoff=10e6, fs=sampling_frequency)

# Apply the zero-phase moving average filter
df['Moving Average'] = moving_average_filter(df['Baseline Subtracted'], window_size=12)

print("\n--- DataFrame with all filtered data ---")
display(df.head())

# Cell 3: Corrected Baseline & Noise Diagnostic Plots

# --- 1. Analyze the UNFILTERED pre-trigger region ---
unfiltered_pre_trigger_data = df['Baseline Subtracted'][:pre_trigger_samples]
unfiltered_noise_sigma = unfiltered_pre_trigger_data.std()

# --- 2. Analyze the FILTERED pre-trigger region ---
# We use .dropna() in case the first few values are NaN from filtering
filtered_pre_trigger_data = df['Moving Average'].dropna()[:pre_trigger_samples]
filtered_noise_sigma = filtered_pre_trigger_data.std() # This is the value we need!

print(f"--- Noise Analysis ---")
print(f"Unfiltered Noise Level (σ_raw): {unfiltered_noise_sigma:.2f} ADC counts")
print(f"Filtered Noise Level (σ_filt): {filtered_noise_sigma:.2f} ADC counts")
print(f"-> The filter reduced the noise by a factor of {unfiltered_noise_sigma/filtered_noise_sigma:.1f}x")


# --- Diagnostic Plot 1: Pre-trigger Histogram (This plot remains the same) ---
fig_hist = go.Figure()
fig_hist.add_trace(go.Histogram(x=unfiltered_pre_trigger_data, name='Pre-trigger Noise'))
fig_hist.update_layout(
    title_text="<b>Diagnostic 1:</b> Histogram of Unfiltered Pre-Trigger Noise",
    xaxis_title="ADC Counts (after baseline subtraction)",
    yaxis_title="Frequency"
)
fig_hist.show()


# --- Diagnostic Plot 2: Zoomed-in view using CORRECTED noise level ---
fig_zoom = go.Figure()

# Plot the moving average trace for the pre-trigger region
fig_zoom.add_trace(go.Scatter(
    x=df['Time (ns)'][:pre_trigger_samples],
    y=df['Moving Average'][:pre_trigger_samples],
    name='Filtered Waveform (Noise)',
    mode='lines',
    line=dict(color='crimson')
))

# Add lines showing the CORRECT, filtered noise level (+/- 3σ)
fig_zoom.add_hline(y=3 * filtered_noise_sigma, line_dash="dash", line_color="orange", annotation_text="3σ (Filtered)")
fig_zoom.add_hline(y=-3 * filtered_noise_sigma, line_dash="dash", line_color="orange")
fig_zoom.add_hline(y=0, line_color="black") # Zero line

fig_zoom.update_layout(
    title_text="<b>Diagnostic 2:</b> Zoomed-in View of Filtered Noise",
    xaxis_title="Time (ns)",
    yaxis_title="Filtered ADC Value"
)
fig_zoom.show()

#======================================
# Check what waveform should look like
#======================================

import matplotlib.pyplot as plt

with uproot.open(file_path) as file:
    keys = file.keys()
    if waveform_index_to_plot < len(keys):
        wf_key = keys[waveform_index_to_plot]
        waveform_np = file[wf_key].values()
        plt.plot(waveform_np)
        plt.title(f"Waveform #{waveform_index_to_plot}")
        plt.xlabel("Time (samples)")
        plt.ylabel("ADC Counts")
        plt.grid(True)
        plt.show()
    else:
        print(f"Invalid waveform index: {waveform_index_to_plot}")

# Cell 4: Charge-Based SPE Finder

# --- 1. Find all peaks using the adaptive prominence threshold ---
# We still use prominence to LOCATE the peaks.
adaptive_prominence = 3 * filtered_noise_sigma
all_peaks, _ = find_peaks(
    -df['Moving Average'].dropna(),
    prominence=adaptive_prominence
)

# --- 2. Calculate the CHARGE for each found peak ---
peak_charges = []
charge_integration_window = 10 # Integrate +/- 5 samples around the peak

for peak_loc in all_peaks:
    # Define the integration window for this specific peak
    start = max(0, peak_loc - charge_integration_window // 2)
    end = min(len(df), peak_loc + charge_integration_window // 2)

    # Calculate charge by summing the inverted ADC counts in the window
    # We use the baseline-subtracted data for a more accurate charge value.
    charge = -df['Baseline Subtracted'][start:end].sum()
    peak_charges.append(charge)

peak_charges = np.array(peak_charges) # Convert to numpy array

# --- 3. Find the SPE Charge from a histogram ---
if len(peak_charges) > 5:
    # You may need to adjust the 'range' based on your expected charge values
    counts, bin_edges = np.histogram(peak_charges, bins=100, range=(0, 2000))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Find the bin with the most counts to estimate the SPE charge
    spe_charge_bin_index = np.argmax(counts)
    spe_charge_estimate = bin_centers[spe_charge_bin_index]

    print(f"✅ Found {len(peak_charges)} significant peaks.")
    print(f"   Estimated Single Photoelectron (SPE) Charge: {spe_charge_estimate:.2f} (integrated ADC counts)")
else:
    spe_charge_estimate = None
    print(f"⚠️ Not enough significant peaks found to estimate SPE charge.")


# --------------------- Tuned SPE processing for 250 MHz, negative pulses ---------------------
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Sampling parameters
fs = 250e6                    # 250 MHz
dt = 1/fs                     # 4 ns/sample

# Baseline pretrigger region
pre_ns = 500                  # 500 ns pretrigger
pre_samp = int(pre_ns*1e-9*fs)

# Bandpass filter for detection only
bp_lo, bp_hi = 3e5, 30e6
b, a = butter(2, [bp_lo/(fs/2), bp_hi/(fs/2)], btype="band")

# Peak selection parameters
min_width_ns, max_width_ns = 12, 200
min_width = max(1, int(min_width_ns*1e-9*fs))
max_width = int(max_width_ns*1e-9*fs)
prom_sigma = 5.5

# Integration window relative to peak
int_pre_ns, int_post_ns = 8, 120
int_pre  = int(int_pre_ns*1e-9*fs)
int_post = int(int_post_ns*1e-9*fs)

# Pile-up guard
refractory_ns = 200
refractory = int(refractory_ns*1e-9*fs)

def baseline_stats(x, pre=pre_samp):
    pre = min(pre, len(x))
    r = x[:pre]
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826*mad if mad>0 else np.std(r)
    return med, sigma

def detect_peaks(x_raw):
    base, sig = baseline_stats(x_raw)
    x0 = x_raw - base
    xf = filtfilt(b, a, x0, method="gust") if len(x0) > 3*max(1,min_width) else x0
    det = -xf  # negative pulses → invert for detection
    peaks, props = find_peaks(
        det,
        prominence=prom_sigma*sig,
        width=(min_width, max_width)
    )
    if len(peaks) > 1:
        keep = [peaks[0]]
        for p in peaks[1:]:
            if p - keep[-1] >= refractory:
                keep.append(p)
        peaks = np.array(keep, dtype=int)
    return peaks, base, x0

def integrate_charge(x_raw, peak_idx, base):
    x0 = x_raw - base
    a = max(0, peak_idx - int_pre)
    b = min(len(x0), peak_idx + int_post)
    q = np.trapz(x0[a:b])
    return -q  # negative pulses → positive charge

# ---------- MAIN: expects waveforms shape (N, T) ----------
# waveforms = np.array([...], dtype=float)

charges = []
sel_idx = []
for i, x in enumerate(waveforms):
    peaks, base, x0 = detect_peaks(x)
    if len(peaks) != 1:
        continue
    q = integrate_charge(x, peaks[0], base)
    charges.append(q)
    sel_idx.append(i)

charges = np.asarray(charges)
print(f"Kept {len(charges)} single-peak events out of {len(waveforms)}")

# Fit pedestal + 1pe with GMM
X = charges.reshape(-1,1)
gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0).fit(X)
means = np.sort(gmm.means_.flatten())
covs  = np.sort(gmm.covariances_.flatten())
ped_mu, one_mu = means[0], means[1]
one_sigma = np.sqrt(covs[1])
S = (one_mu - ped_mu)/max(one_sigma, 1e-12)
print(f"Pedestal ≈ {ped_mu:.4g}, 1pe ≈ {one_mu:.4g}, S ≈ {S:.2f}")

plt.figure()
plt.hist(charges, bins=200)
plt.xlabel("Integrated charge (ADC·sample)")
plt.ylabel("Counts")
plt.title("Charge spectrum (exactly-one-peak events)")
plt.show()
