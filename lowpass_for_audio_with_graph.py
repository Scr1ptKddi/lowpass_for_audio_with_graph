#Step0: import the packages------------------------------------------------------------------------

import warnings
import numpy as np
from scipy import fftpack
from scipy import signal
from scipy.io.wavfile import read, write
from scipy.io.wavfile import WavFileWarning
from matplotlib import pyplot as plt

#ignore WavFileWarning
warnings.simplefilter("ignore", WavFileWarning)

#--------------------------------------------------------------------------------------------------

#Step1: load the data and adjust it for filtering--------------------------------------------------

#Users enter the input and output file path
input_wav_path = input("Input wav file path (e.g., 'input.wav'): ")
output_wav_path = input("Output wav file path (e.g., 'filteref_output.wav': " )

#load the wav file, and save the original data for future plotting
samplerate, data = read(input_wav_path)

#check the datatype
print(f"Original Data Type: {data.dtype}, Min: {data.min()}, Max: {data.max()}")

#if the data is int16, turn it into float 64 for filtering
if data.dtype == np.int16:
    original_dtype = 16
    print("Converting int16 data to float64 for filtering.")
    data = data.astype(np.float64)
    #scaling the amplitude from -1 to 1
    data /= np.iinfo(np.int16).max

#do the same for int32
elif data.dtype == np.int32:
    original_dtype = 32
    print("Converting int32 data to float64 for filtering.")
    data = data.astype(np.float64)
    data /= np.iinfo(np.int32).max

#if the data is stereo, turn it into monaural
if data.ndim > 1:
    print("Stereo data detected.  Using only the left channel.")
    data = data[:, 0]

#check the length of the data
print(f"Data length after conersion: {len(data)}")
#--------------------------------------------------------------------------------------------------

#Step2: Set the parameter for lowpass filtering----------------------------------------------------

x = np.arange(0, len(data)) / samplerate
fp = 23000
fs = 24000
gpass = 0.1
gstop = 30

print(f"samplerate is: {samplerate}")
#--------------------------------------------------------------------------------------------------

#Step3: Applying the lowpass filter----------------------------------------------------------------

#Butterworth Filter (Lowpass)
def lowpass(x, samplerate, fp, fs, gpass, gstop):
    fn = samplerate / 2
    wp = fp / fn
    ws = fs / fn
    N, Wn = signal.buttord(wp, ws, gpass, gstop)
    b, a = signal.butter(N, Wn, "low")
    y = signal.filtfilt(b, a, x)
    return y

data_filt = lowpass(data, samplerate, fp, fs, gpass, gstop)
#--------------------------------------------------------------------------------------------------

#Step4: Performing averaging Fourier transform and decibel conversion------------------------------

#overlap processing
Fs  = 4096
overlap = 90
def ov(data, samplerate, Fs, overlap):
    Ts = len(data) / samplerate
    Fc = Fs / samplerate
    x_ol = Fs * (1 - (overlap / 100))
    N_ave = int((Ts - (Fc * (overlap / 100))) / (Fc * (1 - (overlap / 100))))
    
    array = []

    #extract data
    for i in range(N_ave):
        ps = int(x_ol * i)
        array.append(data[ps:ps + Fs:1])
    return array, N_ave

#Window function processing (Hanning type)
def hanning(data_array, Fs, N_ave):
    han = signal.windows.hann(Fs)
    acf = 1 / (sum(han) / Fs)

    #process all of the overlapped waves with hanning window function
    for i in range(N_ave):
        data_array[i] = data_array[i] * han

    return data_array, acf

#FFT processing
def fft_ave(data_array, samplerate, Fs, N_ave, acf):
    fft_array = []
    for i in range(N_ave):
        fft_array.append(acf * np.abs(fftpack.fft(data_array[i]) / (Fs / 2)))

    fft_axis = np.linspace(0, samplerate, Fs)
    fft_array = np.array(fft_array)
    fft_mean = np.mean(fft_array, axis = 0)

    return fft_array, fft_mean, fft_axis

#turn linear component into dB
def db(x, dBref):
    y = 20 * np.log10(x / dBref)
    return y

#execute the function: overlapped timewave_array
t_array_org, N_ave_org = ov(data, samplerate, Fs, overlap)
t_array_filt, N_ave_filt = ov(data_filt, samplerate, Fs, overlap)

#execute created funciton:process with hanning window function
t_array_org, acf_org = hanning(t_array_org, Fs, N_ave_org)
t_array_filt, acf_filt = hanning(t_array_filt, Fs, N_ave_filt)

#execute created function:process with FFT
fft_array_org, fft_mean_org, fft_axis_org = fft_ave(t_array_org, samplerate, Fs, N_ave_org, acf_org)
fft_array_filt, fft_mean_filt, fft_axis_filt = fft_ave(t_array_filt, samplerate, Fs, N_ave_filt, acf_filt)

#turn into dB
fft_mean_org = db(fft_mean_org, 2e-5)
fft_mean_filt = db(fft_mean_filt, 2e-5)
#--------------------------------------------------------------------------------------------------

#Step5: adjusting the scaling----------------------------------------------------------------------

data_filt /= np.max(np.abs(data_filt))

if original_dtype == 16:
    data_filt *= np.iinfo(np.int16).max
    #turn the datatype back to int16
    data_filt = data_filt.astype(np.int16)

elif original_dtype == 32:
    data_filt *= np.iinfo(np.int32).max
    #turn the datatype back to int32
    data_filt = data_filt.astype(np.int32)
#--------------------------------------------------------------------------------------------------

#Step6: save the filtered data---------------------------------------------------------------------

write(output_wav_path, samplerate, data_filt)

print(f"Filtered audio saved to {output_wav_path}")
#--------------------------------------------------------------------------------------------------

#Step7: plot the graph-----------------------------------------------------------------------------

#set the size and type of the font
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'serif'

#turn the scale inward
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

#apply the scale line to the graph
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.xaxis.set_ticks_position('both')
ax1.yaxis. set_ticks_position('both')
ax2 = fig.add_subplot(212)
ax2.xaxis.set_ticks_position('both')
ax2.yaxis.set_ticks_position('both')

#set the axxis label
ax1.set_xlabel('Time[s]')
ax1.set_ylabel('Amplitude')
ax2.set_xlabel('Frequency[Hz]')
ax2.set_ylabel('SPL[dB]')

#prepare the data plot, and add labels, widths, explanatory notes
ax1.plot(x, data, label='original', lw=1)
ax1.plot(x, data_filt, label='filtered', lw=1)
ax2.plot(fft_axis_org, fft_mean_org, label='original_fft', lw=1)
ax2.plot(fft_axis_filt, fft_mean_filt, label='filtered_fft', lw=1)

#set the limit of the axis
ax2.set_xlim(0, max(fft_axis_org)/2)
ax2.set_ylim(-20, 100)

#set the layout
fig.tight_layout()


#Add legends for clarify
ax1.legend()
ax2.legend()

#Also, plot the graph for difference
difference = data - data_filt
plt.figure()
plt.plot(x, difference, label='Difference', lw=1)
plt.xlabel('Time[s]')
plt.ylabel('Amplitude Difference')
plt.legend()

#show the graph
plt.show()
#--------------------------------------------------------------------------------------------------

