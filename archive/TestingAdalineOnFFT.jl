using WAV
using Plots
using Random
using LinearAlgebra
using SignalAnalysis

mutable struct Adaline
	bias::Float64
	weights::Vector{Float64}
	learning_rate::Float64
	epochs::Int
	loss_history::Vector{Float64}

	function Adaline(input_size::Union{Int, Tuple}; learning_rate = 0.001, epochs = 100)
		bias = randn()
		weights = randn(input_size)
		new(bias, weights, learning_rate, epochs)
	end
end

# Holds the features of a Short Term Fourier Transforamtion
mutable struct STFTModel
	mag::Union{Matrix{Float64}, Matrix{ComplexF64}} #Magnitudes
	time::Union{Nothing, AbstractRange{Float64}, Vector{Float64}}
	freq::Union{Nothing, AbstractRange{Float64}, Vector{Float64}}
	n::Union{Int, Nothing}
	noverlap::Union{Int, Nothing}
	nfft::Union{Int, Nothing}
	fs::Union{Int, Float64}
	window::Union{AbstractVector, Function, Nothing}

	function STFTModel(
		mag::Union{Matrix{Float64}, Matrix{ComplexF64}},
		time::Union{Nothing, AbstractRange{Float64}, Vector{Float64}} = nothing,
		freq::Union{Nothing, AbstractRange{Float64}, Vector{Float64}} = nothing;
		n::Union{Int, Nothing} = nothing,
		noverlap::Union{Int, Nothing} = nothing,
		nfft::Union{Int, Nothing} = nothing,
		fs::Union{Int, Float64} = 48000.0,
		window::Union{AbstractVector, Function, Nothing} = nothing,
	)
		new(mag, time, freq, n, noverlap, nfft, fs, window)
	end
end

function height(model::STFTModel)
	return model.freq === nothing ? size(model.mag, 1) : length(model.freq)
end

function width(model::STFTModel)
	return model.time === nothing ? size(model.mag, 2) : length(model.time)
end

function preprocess_audio(file_path::String, sr = 48000)
	aud, fs = wavread(file_path)

	y = length(aud[1, :])
	x = length(aud) ÷ y

	# Choosing the first array of the stereo matrix
	aud = aud[:, 1]

	# Matching the data rate
	if fs != sr
		ratio = sr / fs
		aud = resample(aud, ratio)
	end

	# Normalization: range (-1,1)
	aud = aud / maximum(abs.(aud))
	# aud_min, aud_max = extrema(aud)
	# if aud_max > aud_min
	# 	aud = @. (aud - aud_min) / (aud_max - aud_min) * 2 - 1
	# end
	# Ensure the audio is in the range [-1, 1]

	return aud, sr
end

function toSTFT(wav, n = 16000, noverlap = 12000; nfft = nextfastfft(n), fs = 48000, window = hamming(n))
	# Compute STFT
	# n: Window size (65536 samples)
	# noverlap: Overlap (65280 samples, i.e., 256 samples hop size)
	mags = stft(wav, n, noverlap; nfft, fs, window)
	# Potential Alternatives:
	# S = stft(wav, n, noverlap; fs, window)
	# S = stft(wav, n = div(length(s), 8), noverlap = div(n, 2); onesided = eltype(s)<:Real, nfft = nextfastfft(n), fs = 1, window = nothing)
	# S = stft(aud, n, noverlap; onesided = eltype(aud)<:Real, nfft = nextfastfft(n), fs = fs)

	# Create time and frequency axes
	n_freq = size(mags, 1)
	n_time = size(mags, 2)
	freq = (0:(n_freq-1)) * (fs / n)  # Frequencies: 0 to fs/2
	time = (0:(n_time-1)) * (n - noverlap) / fs  # Time points in seconds

	println("The size of this STFT model is: ", size(mags))
	println("Time range of this model: ", time)
	println("Frequency range of this model: ", freq)
	println()
	return STFTModel(mags, time, freq; n, noverlap, nfft, fs, window)
end

function toWav(stftmodel::STFTModel)
	magnitudes = stftmodel.mag
	nfft = stftmodel.nfft
	noverlap = stftmodel.noverlap
	window = stftmodel.window
	wav = istft(Float64, magnitudes; nfft, noverlap, window)
	return wav
end

function heatmapSTFT(stftmodel::STFTModel; in_db::Bool = false)
	# Prepare magnitude spectrogram
	mag = abs.(stftmodel.mag)

	# Convert to decibel if it is requested
	if in_db
		mag = amp2db.(mag .+ floatmin()) # Add small constant to avoid log(0)
	end

	return heatmap(stftmodel.time, stftmodel.freq, mag,
		title = "STFT Magnitude Spectrogram",
		xlabel = "Time (s)",
		ylabel = "Frequency (Hz)",
		clim = (0, maximum(mag)),  # Adjust color limits
		background = "gray10",
	)
end

function pad(matrix::Matrix{T}, pad_size::Tuple{Int, Int}) where {T <: Any}
	a, b = pad_size
	@assert a >= 0 && b >= 0 "Padding sizes must be non-negative"

	N, M = size(matrix)
	padded_matrix = zeros(T, N + 2 * a, M + 2 * b)
	padded_matrix[(a+1):(a+N), (b+1):(b+M)] = matrix

	return padded_matrix
end

################################################################################


function activation(adaline::Adaline, x::Float64)
	return x
end

function predict(adaline::Adaline, x::Vector{Float64})
	return activation(adaline, dot(x, adaline.weights) + adaline.bias)
end

function denoise_frame(adaline::Adaline, noisy_frame::Vector{Float64}, context_size::Int)
	# Simplified - use entire window
	predict(adaline, noisy_frame)
end

function train!(adaline::Adaline, X::Matrix{Float64}, y::Matrix{Float64})
	loss_history = Float64[]

	n_samples = size(X, 1)
	for ep in 1:adaline.epochs
		total_loss = 0.0
		error = 0
		for i in 1:n_samples
			xi = X[i, :]
			target = y[i]
			prediction = predict(adaline, xi)
			error = target - prediction
			total_loss += error^2
			adaline.weights .+= adaline.learning_rate * error * xi * 2
			adaline.bias += adaline.learning_rate * error * 2
		end
		push!(loss_history, total_loss / n_samples)
	end
	adaline.loss_history = loss_history
	return loss_history
end

function create_training_data(noisy_stft::STFTModel, clean_stft::STFTModel, window_size::Tuple{Int, Int})
	@assert size(clean_stft.mag) == size(noisy_stft.mag)

	# Prepare  padding
	pad_size = window_size .÷ 2
	padded_noisy = pad(noisy_stft.mag, pad_size)

	# Create training matrix
	n_samples = height(noisy_stft) * width(noisy_stft)
	window_height, window_width = window_size
	X = zeros(Float64, 2, n_samples, window_height * window_width)
	for i in 1:height(noisy_stft)
		for j in 1:width(noisy_stft)
			x = padded_noisy[i:(i+window_height-1), j:(j+window_width-1)]
			X[1, (i-1)*width(noisy_stft)+j, :] = real(x)
			X[2, (i-1)*width(noisy_stft)+j, :] = imag(x)
		end
	end

	return X, clean_stft.mag
end

function train_an_adaline_for_denoising(noisy_audio::String, clean_audio::String; window_size::Tuple{Int, Int} = (5, 5), learning_rate = 0.0000001, epochs = 50)
	# Read and normalize audio file
	clean_wav, sr = preprocess_audio(clean_audio)
	noisy_wav, _ = preprocess_audio(noisy_audio)

	# Transform waveform audio to Short-Time Fourier model
	noisy_stft = toSTFT(noisy_wav)
	clean_stft = toSTFT(clean_wav)
	# noisy_stft = toSTFT(noisy_wav)
	# clean_stft = toSTFT(clean_wav)

	X, y = create_training_data(noisy_stft, clean_stft, window_size)
	adaline_rl = Adaline(window_size[1]*window_size[2]; learning_rate = learning_rate, epochs = epochs)
	adaline_im = Adaline(window_size[1]*window_size[2]; learning_rate = learning_rate, epochs = epochs)
	loss_history_rl = train!(adaline_rl, X[1, :, :], real(y))
	loss_history_im = train!(adaline_im, X[2, :, :], imag(y))
	return adaline_rl, adaline_im, loss_history_rl, loss_history_im
end

function denoise_audio(noisy_audio::String, adaline_rl::Adaline, adaline_im::Adaline, output_audio::String, window_size::Tuple{Int, Int} = (5, 5), context_size = 2)
	noisy_wav, fs = preprocess_audio(noisy_audio)

	noisy_stft = toSTFT(noisy_wav)
	# noisy_stft = toSTFT(noisy_wav)

	# Prepare  padding
	pad_size = window_size .÷ 2
	padded_noisy = pad(noisy_stft.mag, pad_size)



	# Create input matrix
	N = width(noisy_stft) * height(noisy_stft)
	# X = zeros(Float64, N, window_size)
	# for i in 1:N
	# 	X[i, :] = padded_noisy[i:(i+window_size-1)]
	# end
	window_height, window_width = window_size
	X = zeros(Float64, 2, N, window_height * window_width)
	for i in 1:height(noisy_stft)
		for j in 1:width(noisy_stft)
			x = padded_noisy[i:(i+window_height-1), j:(j+window_width-1)]
			X[1, (i-1)*width(noisy_stft)+j, :] = real(x)
			X[2, (i-1)*width(noisy_stft)+j, :] = imag(x)
		end
	end


	N, M = height(noisy_stft), width(noisy_stft)
	denoised_rl = zeros(ComplexF64, N, M)
	for i in 1:N
		for j in 1:M
			window = X[1, (i-1)*width(noisy_stft)+j, :]
			denoised_rl[i, j] = denoise_frame(adaline_rl, window, context_size)
		end
	end
	denoised_im = zeros(ComplexF64, N, M)
	for i in 1:N
		for j in 1:M
			window = X[2, (i-1)*width(noisy_stft)+j, :]
			denoised_im[i, j] = denoise_frame(adaline_im, window, context_size)
		end
	end

	denoised_stft = noisy_stft
	denoised_stft.mag = denoised_rl+(denoised_im)im
	denoised_wav = toWav(denoised_stft)

	#????????????????????????
	#!!!!!!!!!!!!!!!!!!!!!!!
	#**********************
	denoised_stft.mag*=2
	display(heatmapSTFT(denoised_stft))

	# Normalize
	denoised_wav = denoised_wav / maximum(abs.(denoised_wav))

	# Save the output
	wavwrite(denoised_wav, output_audio; Fs = fs)
	return denoised_wav, fs
end



function denoise_audio(window_size::Union{Int, Tuple{Int, Int}} = 5; ver::String = "")
	# Add one to window_size if it's even
	window_size = window_size .÷ 2 .* 2 .+ 1

	printstyled("Loading files...\n"; color = :yellow)
	noisy_audio = "./aud/learn/noisy/noisy_audio.wav" #> to train
	clean_audio = "./aud/learn/clean/clean_audio.wav" #> to train
	input_audio = "./aud/input/noisy_test.wav" #> to denoise
	output_audio = "./aud/output/denoised_audio_v$ver.wav" #> to save

	# * input_audio = "./aud/learn/noisy/noisy_audio.wav"
	# * output_audio = "./aud/output/denoised_audio.wav"

	# Verify files exist
	for f in [clean_audio, noisy_audio, input_audio]
		isfile(f) || error("File not found: $f")
	end

	# Define and train an adaline for denoising
	printstyled("Training started...\n"; color = :magenta)
	adaline_rl, adaline_im, lh_rl, lh_im = train_an_adaline_for_denoising(noisy_audio, clean_audio; window_size)

	# Denoising through the trained adaline
	printstyled("Denoising audio...\n"; color = :magenta)
	denoise_audio(input_audio, adaline_rl, adaline_im, output_audio, window_size)
	printstyled("Denoised audio saved to $output_audio\n"; color = :light_green)

	# Optional: Plot training loss
	display(plot(lh_rl+lh_im, title = "Training Loss", xlabel = "Epoch", ylabel = "Mean Squared Error", lw = 2, background = "gray10"))
	lossimglog = "./log/loss_plot_v$ver.png"
	savefig(lossimglog)
	printstyled("An image of training loss saved to $lossimglog\n"; color = :green)
	display(adaline_rl)
	display(adaline_im)
	display(sum(adaline_rl.weights))
	display(sum(adaline_im.weights))
end


# noisy_audio = "./aud/learn/noisy/noisy_audio.wav" #> to train
# clean_audio = "./aud/learn/clean/clean_audio.wav" #> to train
# a, b = train_an_adaline_for_denoising(noisy_audio, clean_audio)
# println(a)
# println(b)

denoise_audio((6, 6); ver = "2.0")

