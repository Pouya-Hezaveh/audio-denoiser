# Checks and installs packages
include("PackageInstaller.jl")
ensure_packages(
	"DSP",
	"WAV",
	"Plots",
	"Random",
	"LinearAlgebra",
)

using DSP
using WAV
using Plots
using Random
using LinearAlgebra

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

function activation(adaline::Adaline, x::Float64)
	return x
end

function predict(adaline::Adaline, x::Vector{Float64})
	return activation(adaline, dot(x, adaline.weights) + adaline.bias)
end

function pad(vector::Vector, pad_size::Int)
	N = length(vector)
	padded_vector = Vector{Float64}(undef, N + 2 * pad_size)
	padded_vector[(pad_size+1):(pad_size+N)] = vector

	# Reflect audio to fill sides of the pad with real audio data
	for i in 1:pad_size
		# Reflect audio on the left side of the pad
		padded_vector[pad_size+1-i] = vector[i+1]
		# Reflect audio on the write side of the pad
		padded_vector[pad_size+N+i] = vector[N-i]
	end

	return padded_vector
end

function train!(adaline::Adaline, X::Matrix{Float64}, y::Vector{Float64})
	loss_history = Float64[]

	n_samples = size(X, 1)
	for _ in 1:adaline.epochs
		total_loss = 0.0
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

function denoise_frame(adaline::Adaline, noisy_frame::Vector{Float64}, context_size::Int)
	# Simplified - use entire window
	predict(adaline, noisy_frame)
end

function create_training_data(noisy_wav::Vector{Float64}, clean_wav::Vector{Float64}, window_size::Int)
	@assert length(clean_wav) == length(noisy_wav)

	# Proper symmetric padding
	pad_size = window_size .÷ 2
	padded_noisy = pad(noisy_wav, pad_size)

	# Create training matrix 
	n_samples = length(clean_wav)
	X = zeros(Float64, n_samples, window_size)
	for i in 1:n_samples
		X[i, :] = padded_noisy[i:(i+window_size-1)]
	end

	return X, clean_wav
end

function train_an_adaline_for_denoising(noisy_audio::String, clean_audio::String; window_size::Int = 5, learning_rate = 0.00001, epochs = 50)
	if (window_size%2==0)
		window_size+=1
	end
	clean_wav, sr = preprocess_audio(clean_audio)
	noisy_wav, _ = preprocess_audio(noisy_audio)
	X, y = create_training_data(noisy_wav, clean_wav, window_size)
	adaline = Adaline(window_size; learning_rate = learning_rate, epochs = epochs)
	loss_history = train!(adaline, X, y)
	return adaline, loss_history
end

function denoise_audio(noisy_audio::String, adaline::Adaline, output_audio::String; context_size = 2)
	noisy_audio, fs = preprocess_audio(noisy_audio)
	window_size = length(adaline.weights)
	pad_size = window_size ÷ 2

	padded_noisy = pad(noisy_audio, pad_size)

	N = length(noisy_audio)
	X = zeros(Float64, N, window_size)
	for i in 1:N
		X[i, :] = padded_noisy[i:(i+window_size-1)]
	end

	denoised = zeros(N)
	for i in 1:N
		window = X[i, :]
		denoised[i] = denoise_frame(adaline, window, context_size)
	end
	# Normalize denoised audio
	denoised = denoised / maximum(abs.(denoised))
	wavwrite(denoised, output_audio; Fs = fs)
	return denoised, fs
end

function main(window_size::Int = 25)
	printstyled("Loading files...\n"; color = :yellow)
	noisy_audio = "./aud/learn/noisy/noisy_audio.wav" #> to train
	clean_audio = "./aud/learn/clean/clean_audio.wav" #> to train

	# input_audio = "./aud/input/noisy_test.wav" #> to denoise
	input_audio = "./aud/input/pilot.wav" #> to denoise
	output_audio = "./aud/output/denoised_audio.wav" #> to save

	# * input_audio = "./aud/learn/noisy/noisy_audio.wav"
	# * output_audio = "./aud/output/denoised_audio.wav"

	# Verify files exist
	for f in [clean_audio, noisy_audio, input_audio]
		isfile(f) || error("File not found: $f")
	end

	# Define and train an adaline for denoising
	printstyled("Training started...\n"; color = :magenta)
	adaline, loss_history = train_an_adaline_for_denoising(noisy_audio, clean_audio; window_size)

	# Denoising through the trained adaline
	printstyled("Denoising audio...\n"; color = :magenta)
	denoise_audio(input_audio, adaline, output_audio)
	printstyled("Denoised audio saved to $output_audio\n"; color = :light_green)

	# Optional: Plot training loss
	display(plot(loss_history, title = "Training Loss", xlabel = "Epoch", ylabel = "Mean Squared Error", lw = 2, background = "gray10"))
	mkpath("log")
	lossimglog = "./log/loss_plot.png"
	savefig(lossimglog)
	printstyled("An image of training loss saved to $lossimglog\n"; color = :green)
end

main()
