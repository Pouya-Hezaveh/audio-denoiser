import librosa
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

# یک فایل صوتی با پسوند WAV را به همراه فایل برنامه در داخل یک پوشه ذخیره و برنامه را اجرا کنید

def add_noise_to_audio(input_file, output_file, noise_type='white', noise_level=0.1):
    """
    Add noise to an audio file and save the result
    
    Parameters:
        input_file (str): Path to input WAV file
        output_file (str): Path to save noisy output WAV
        noise_type (str): Type of noise to add ('white', 'pink', 'brown', 'gaussian')
        noise_level (float): Amplitude of noise (0-1)
    """
    # Load the audio file
    audio, sr = librosa.load(input_file, sr=None)
    
    # Generate noise
    if noise_type == 'white':
        noise = np.random.normal(0, 1, len(audio))
    elif noise_type == 'pink':
        # Pink noise (1/f noise)
        uneven = len(audio) % 2
        x = np.random.normal(0, 1, len(audio) // 2 + 1 + uneven)
        f = np.fft.rfft(x)
        f[1:] /= np.sqrt(np.arange(1, len(f)))[:]
        noise = np.fft.irfft(f)[:len(audio)]
    elif noise_type == 'brown':
        # Brown noise (1/f^2 noise)
        uneven = len(audio) % 2
        x = np.random.normal(0, 1, len(audio) // 2 + 1 + uneven)
        f = np.fft.rfft(x)
        f[1:] /= np.arange(1, len(f))
        noise = np.fft.irfft(f)[:len(audio)]
    elif noise_type == 'gaussian':
        noise = np.random.normal(0, noise_level, len(audio))
    else:
        raise ValueError(f"Unknown noise type: {noise_type}")
    
    # Normalize the noise
    noise = noise / np.max(np.abs(noise)) * noise_level
    
    # Mix audio and noise
    noisy_audio = audio + noise
    
    # Normalize to prevent clipping
    noisy_audio = noisy_audio / np.max(np.abs(noisy_audio))
    
    # Save the noisy audio
    sf.write(output_file, noisy_audio, sr)
    
    # Plot for visualization
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.title("Original Audio")
    plt.plot(audio)
    plt.ylim(-1, 1)
    
    plt.subplot(3, 1, 2)
    plt.title(f"{noise_type.capitalize()} Noise")
    plt.plot(noise)
    plt.ylim(-1, 1)
    
    plt.subplot(3, 1, 3)
    plt.title("Noisy Audio")
    plt.plot(noisy_audio)
    plt.ylim(-1, 1)
    
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    input_file = "clean_audio.wav"  # Replace with your file
    output_file = "noisy_audio.wav"
    
    # Available noise types: 'white', 'pink', 'brown', 'gaussian'
    add_noise_to_audio(
        input_file=input_file,
        output_file=output_file,
        noise_type='white',  # Try different noise types
        noise_level=0.2      # Adjust noise level (0.1-0.5 is usually good)
    )
    
    print(f"Added noise to {input_file} and saved as {output_file}")
    
