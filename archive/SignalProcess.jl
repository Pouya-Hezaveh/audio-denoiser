using WAV
using Plots
using SignalAnalysis

function preprocess_audio(file_path::String; sr = 48000)
	aud, fs = wavread(file_path)

	y = length(aud[1, :])
	x = length(aud) ÷ y

	# Choosing the first array of the stereo matrix
	aud = aud[:,1]

	# Matching the data rate
	if fs != sr
		ratio = sr / fs
		aud = resample(aud, ratio)
	end

	#=
	# Normalization
	aud_min, aud_max = extrema(aud)
	if aud_max > aud_min
		aud = @. (aud - aud_min) / (aud_max - aud_min) * 2 - 1
	end
	=#
	return aud, sr
end


# Load the WAV file
aud, fs = preprocess_audio("./aud/input/audio_to_denoise.wav")


#=
spec = spectrogram(aud; fs = fs)

# Plot the spectrogram
heatmap(spec.time, spec.freq, spec.power, xlabel = "X-axis", ylabel = "Y-axis", title = "Heatmap")

sg = stft(aud, n, noverlap; onesided = eltype(aud)<:Real, nfft = nextfastfft(n), fs = fs)
=#####################################

# Computes The Short-Time Fourier transform
function getSTFT(aud)
	n = 2^16          # Window size
	noverlap = 2^16-2^8   # Overlap
	window = hamming(n)  # Hamming window
	S = stft(aud, n, noverlap; fs = fs, window = window)
	
	# Prepare magnitude spectrogram
	magnitude = abs.(S)
	# magnitude = 20 * log10.(abs.(S) .+ 1e-10)  # Add small constant to avoid log(0)

	# Create time and frequency axes
	n_freq = size(S, 1)
	n_time = size(S, 2)
	freq = (0:(n_freq-1)) * (fs / n)  # Frequencies: 0 to fs/2 (24000 Hz)
	time = (0:(n_time-1)) * (n - noverlap) / fs  # Time points in seconds

	return time, freq, magnitude
end

time,freq,magnitude = getSTFT(aud)
println("Ajab!")

# Plot heatmap
heatmap(time, freq, magnitude,
	title = "STFT Magnitude Spectrogram",
	xlabel = "Time (s)",
	ylabel = "Frequency (Hz)",
	color = :viridis,
	clim = (0, maximum(magnitude)),  # Adjust color limits
	size = (800, 600),
)




