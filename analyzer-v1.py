import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import spectrogram, butter, filtfilt, find_peaks
from scipy import fft
import soundfile as sf
import os

def analyze_grid_frequency(file_path, plot_results=True):
    """
    Analyze an audio recording of an electrical arc to determine the grid frequency.
    
    Parameters:
    file_path (str): Path to the audio file (.ogg or other format)
    plot_results (bool): Whether to display visualization plots
    
    Returns:
    float: Detected grid frequency in Hz
    """
    print(f"Analyzing file: {file_path}")
    
    # Load the audio file
    try:
        data, sample_rate = sf.read(file_path)
        print(f"File loaded. Sample rate: {sample_rate} Hz, Duration: {len(data)/sample_rate:.2f} seconds")
    except Exception as e:
        print(f"Error loading file: {e}")
        return None
    
    # Convert stereo to mono if needed
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = np.mean(data, axis=1)
        print("Converted stereo to mono")
    
    # Normalize the audio
    data = data / np.max(np.abs(data))
    
    # Apply a bandpass filter (40-70 Hz range to focus on grid frequencies)
    def butter_bandpass(lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a
    
    def bandpass_filter(data, lowcut, highcut, fs, order=5):
        b, a = butter_bandpass(lowcut, highcut, fs, order=order)
        y = filtfilt(b, a, data)
        return y
    
    # Filter settings for grid frequency (40-70 Hz covers both 50Hz and 60Hz grids)
    filtered_data = bandpass_filter(data, 40, 70, sample_rate, order=3)
    print("Applied bandpass filter (40-70 Hz)")
    
    # Create a spectrogram
    frequencies, times, Sxx = spectrogram(filtered_data, sample_rate, nperseg=int(sample_rate*0.5))
    
    # Calculate the FFT of the filtered signal
    n = len(filtered_data)
    yf = fft.rfft(filtered_data)
    xf = fft.rfftfreq(n, 1 / sample_rate)
    
    # Only look at the relevant frequency range (40-70 Hz)
    mask = (xf >= 40) & (xf <= 70)
    xf_masked = xf[mask]
    yf_masked = np.abs(yf[mask])
    
    # Find peaks in the frequency domain
    peaks, _ = find_peaks(yf_masked, height=0.05*np.max(yf_masked))
    
    if len(peaks) == 0:
        print("No significant peaks found in the expected frequency range (40-70 Hz)")
        grid_freq = None
    else:
        # Get the highest peak
        main_peak = peaks[np.argmax(yf_masked[peaks])]
        grid_freq = xf_masked[main_peak]
        print(f"Detected grid frequency: {grid_freq:.2f} Hz")
        
        # Find all significant peaks
        print("Significant frequency peaks found:")
        for peak in peaks:
            freq = xf_masked[peak]
            amplitude = yf_masked[peak]
            print(f"  {freq:.2f} Hz (amplitude: {amplitude:.4f})")
    
    # Plot the results if requested
    if plot_results:
        plt.figure(figsize=(15, 10))
        
        # Plot original waveform
        plt.subplot(3, 1, 1)
        time_axis = np.arange(len(data)) / sample_rate
        plt.plot(time_axis, data)
        plt.title('Original Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot filtered waveform
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, filtered_data)
        plt.title('Filtered Waveform (40-70 Hz Bandpass)')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot frequency spectrum
        plt.subplot(3, 1, 3)
        plt.plot(xf_masked, yf_masked)
        if len(peaks) > 0:
            plt.plot(xf_masked[peaks], yf_masked[peaks], "x", color='red')
            for peak in peaks:
                plt.text(xf_masked[peak], yf_masked[peak], f"{xf_masked[peak]:.2f} Hz")
        plt.title('Frequency Spectrum (40-70 Hz Range)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(40, 70)
        
        plt.tight_layout()
        plt.savefig('grid_frequency_analysis.png')
        plt.show()
    
    return grid_freq

def main():
    """Main function to run the analysis interactively"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze audio to detect grid frequency.')
    parser.add_argument('file_path', type=str, help='Path to the audio file')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    
    args = parser.parse_args()
    
    analyze_grid_frequency(args.file_path, plot_results=not args.no_plot)

if __name__ == "__main__":
    main()
