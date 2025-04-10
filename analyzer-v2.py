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
    
    # Apply a bandpass filter to focus on potential arc frequencies
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
    
    # Create two filtered datasets - one for fundamental grid frequency and one for the acoustic signature
    filtered_data_fundamental = bandpass_filter(data, 40, 70, sample_rate, order=3)
    filtered_data_acoustic = bandpass_filter(data, 80, 140, sample_rate, order=3)
    print("Applied bandpass filters (40-70 Hz and 80-140 Hz)")
    
    # Calculate the FFT of both filtered signals
    n = len(data)
    
    # For fundamental frequency range (40-70 Hz)
    yf_fundamental = fft.rfft(filtered_data_fundamental)
    xf = fft.rfftfreq(n, 1 / sample_rate)
    mask_fundamental = (xf >= 40) & (xf <= 70)
    xf_fund_masked = xf[mask_fundamental]
    yf_fund_masked = np.abs(yf_fundamental[mask_fundamental])
    
    # For acoustic frequency range (80-140 Hz)
    yf_acoustic = fft.rfft(filtered_data_acoustic)
    mask_acoustic = (xf >= 80) & (xf <= 140)
    xf_acoustic_masked = xf[mask_acoustic]
    yf_acoustic_masked = np.abs(yf_acoustic[mask_acoustic])
    
    # Find peaks in both frequency domains
    fund_peaks, _ = find_peaks(yf_fund_masked, height=0.05*np.max(yf_fund_masked))
    acoustic_peaks, _ = find_peaks(yf_acoustic_masked, height=0.05*np.max(yf_acoustic_masked))
    
    # Process fundamental frequency range (40-70 Hz)
    if len(fund_peaks) == 0:
        print("No significant peaks found in the fundamental frequency range (40-70 Hz)")
        fund_freq = None
    else:
        # Get the highest peak
        main_peak = fund_peaks[np.argmax(yf_fund_masked[fund_peaks])]
        fund_freq = xf_fund_masked[main_peak]
        print(f"Detected possible fundamental grid frequency: {fund_freq:.2f} Hz")
        
        # Find all significant peaks
        print("Significant frequency peaks in fundamental range (40-70 Hz):")
        for peak in fund_peaks:
            freq = xf_fund_masked[peak]
            amplitude = yf_fund_masked[peak]
            print(f"  {freq:.2f} Hz (amplitude: {amplitude:.4f})")
    
    # Process acoustic frequency range (80-140 Hz)
    if len(acoustic_peaks) == 0:
        print("No significant peaks found in the acoustic frequency range (80-140 Hz)")
        acoustic_freq = None
    else:
        # Get the highest peak
        main_peak = acoustic_peaks[np.argmax(yf_acoustic_masked[acoustic_peaks])]
        acoustic_freq = xf_acoustic_masked[main_peak]
        print(f"Detected acoustic signature frequency: {acoustic_freq:.2f} Hz")
        
        # Calculate the fundamental grid frequency (half the acoustic frequency)
        derived_grid_freq = acoustic_freq / 2
        print(f"Derived grid frequency: {derived_grid_freq:.2f} Hz")
        
        # Find all significant peaks
        print("Significant frequency peaks in acoustic range (80-140 Hz):")
        for peak in acoustic_peaks:
            freq = xf_acoustic_masked[peak]
            amplitude = yf_acoustic_masked[peak]
            derived = freq / 2
            print(f"  {freq:.2f} Hz (amplitude: {amplitude:.4f}, derived grid freq: {derived:.2f} Hz)")
    
    # Plot the results if requested
    if plot_results:
        plt.figure(figsize=(15, 12))
        
        # Plot original waveform
        plt.subplot(4, 1, 1)
        time_axis = np.arange(len(data)) / sample_rate
        plt.plot(time_axis, data)
        plt.title('Original Waveform')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        
        # Plot both filtered waveforms
        plt.subplot(4, 1, 2)
        plt.plot(time_axis, filtered_data_fundamental, label='40-70 Hz')
        plt.plot(time_axis, filtered_data_acoustic, label='80-140 Hz', alpha=0.7)
        plt.title('Filtered Waveforms')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.legend()
        
        # Plot fundamental frequency spectrum
        plt.subplot(4, 1, 3)
        plt.plot(xf_fund_masked, yf_fund_masked)
        if len(fund_peaks) > 0:
            plt.plot(xf_fund_masked[fund_peaks], yf_fund_masked[fund_peaks], "x", color='red')
            for peak in fund_peaks:
                plt.text(xf_fund_masked[peak], yf_fund_masked[peak], f"{xf_fund_masked[peak]:.2f} Hz")
        plt.title('Frequency Spectrum (40-70 Hz Range)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(40, 70)
        
        # Plot acoustic frequency spectrum
        plt.subplot(4, 1, 4)
        plt.plot(xf_acoustic_masked, yf_acoustic_masked)
        if len(acoustic_peaks) > 0:
            plt.plot(xf_acoustic_masked[acoustic_peaks], yf_acoustic_masked[acoustic_peaks], "x", color='red')
            for peak in acoustic_peaks:
                plt.text(xf_acoustic_masked[peak], yf_acoustic_masked[peak], 
                         f"{xf_acoustic_masked[peak]:.2f} Hz")
        plt.title('Frequency Spectrum (80-140 Hz Range)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(80, 140)
        
        plt.tight_layout()
        plt.savefig('grid_frequency_analysis.png')
        plt.show()
    
    # Compare results and identify the most likely grid frequency
    if acoustic_freq is not None:
        # If we have a strong acoustic signature, use that to determine grid frequency
        grid_freq = acoustic_freq / 2
        confidence = "high" if acoustic_freq in [100, 120] else "medium"
    elif fund_freq is not None:
        # Fall back to direct measurement if no acoustic signature
        grid_freq = fund_freq
        confidence = "medium" if fund_freq in [50, 60] else "low"
    else:
        grid_freq = None
        confidence = "none"
    
    print(f"\nFinal analysis:")
    if grid_freq is not None:
        print(f"Most likely grid frequency: {grid_freq:.2f} Hz (confidence: {confidence})")
        
        # Check if it's close to standard frequencies
        if abs(grid_freq - 50) < 0.55:
            print("This matches the standard 50 Hz grid (Europe, Asia, Africa, most of South America)")
        elif abs(grid_freq - 60) < 0.55:
            print("This matches the standard 60 Hz grid (North America, parts of South America)")
        else:
            print("This does not match standard grid frequencies - could indicate measurement error or special grid conditions")
    else:
        print("Could not determine grid frequency from the recording")
    
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
