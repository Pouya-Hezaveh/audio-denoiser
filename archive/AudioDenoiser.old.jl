# Checks and installs packages
include("PackageInstaller.jl")
ensure_packages(
	"WAV",
	"DSP",
	"Plots",
	"Random",
	"LinearAlgebra",
)

using WAV
using DSP
using Plots
using Random
using LinearAlgebra

mutable struct Adaline
	bias::Float64
	weights::Vector{Float64}
	learning_rate::Float64
	epochs::Int

	function Adaline(input_size; learning_rate = 0.01, epochs = 50)
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
			adaline.weights .+= adaline.learning_rate * error * xi
			adaline.bias += adaline.learning_rate * error
		end

		push!(loss_history, total_loss / n_samples)
	end

	return loss_history
end

function denoise_frame(adaline::Adaline, noisy_frame::Vector{Float64}, context_size::Int)
	# Simplified - use entire window
	predict(adaline, noisy_frame)
end

function preprocess_audio(file_path::String; sr = 48000)
	aud, fs = wavread(file_path)

	y = length(aud[1, :])
	x = length(aud) ÷ y

	# Combining parallel data of many microphones into an average one
	temp = zeros(x)
	for i in 1:y
		temp += aud[:, i]
	end
	temp/=y
	aud = temp

	# Matching the data rate
	if fs != sr
		ratio = sr / fs
		aud = resample(aud, ratio)
	end

	# Normalization
	aud_min, aud_max = extrema(aud)
	if aud_max > aud_min
		aud = @. -1 + 2 * (aud - aud_min) / (aud_max - aud_min)
	end
	return aud, sr
end

function create_training_data(noisy_audio::Vector{Float64}, clean_audio::Vector{Float64}, window_size::Int)
	@assert length(clean_audio) == length(noisy_audio)

	# Proper symmetric padding
	pad_size = window_size ÷ 2
	padded_noisy = [zeros(pad_size); noisy_audio; zeros(pad_size)]

	# Create training matrix
	n_samples = length(clean_audio)
	X = zeros(Float64, n_samples, window_size)
	for i in 1:n_samples
		X[i, :] = padded_noisy[i:(i+window_size-1)]
	end
	return X, clean_audio
end

function train_an_adaline_for_denoising(noisy_audio::String, clean_audio::String; window_size = 5, learning_rate = 0.001, epochs = 50)
	clean_audio, sr = preprocess_audio(clean_audio)
	noisy_audio, _ = preprocess_audio(noisy_audio)
	X, y = create_training_data(noisy_audio, clean_audio, window_size)
	adaline = Adaline(window_size; learning_rate = learning_rate, epochs = epochs)
	loss_history = train!(adaline, X, y)
	plot(loss_history, title = "Training Loss", xlabel = "Epoch", ylabel = "Mean Squared Error", legend = false)
	Plots.display(plot!())
	return adaline, loss_history
end

function denoise_audio(noisy_audio::String, adaline::Adaline, output_audio::String; context_size = 2)
	noisy_audio, sr = preprocess_audio(noisy_audio)
	window_size = length(adaline.weights)
	pad_size = window_size ÷ 2
	N = length(noisy_audio)
	padded_noisy = Vector{Float64}(undef, N + 2 * pad_size)
	padded_noisy[(pad_size+1):(pad_size+N)] = noisy_audio
	for i in 1:pad_size
		padded_noisy[pad_size+1-i] = noisy_audio[i+1]
	end
	for i in 1:pad_size
		padded_noisy[pad_size+N+i] = noisy_audio[N-i]
	end
	X = zeros(Float64, N, window_size)
	for i in 1:N
		X[i, :] = padded_noisy[i:(i+window_size-1)]
	end
	denoised = zeros(N)
	for i in 1:N
		window = X[i, :]
		denoised[i] = denoise_frame(adaline, window, context_size)
	end
	wavwrite(denoised, output_audio; Fs = sr)
	return denoised, sr
end

function main()
	printstyled("Loading files...\n"; color = :yellow)
	clean_audio = "./aud/learn/clean/clean_audio.wav"
	noisy_audio = "./aud/learn/noisy/noisy_audio.wav"
	input_audio = "./aud/input/noisy_test.wav"
	output_audio = "./aud/output/denoised_audio_old.wav"

	# Verify files exist
	for f in [clean_audio, noisy_audio, input_audio]
		isfile(f) || error("File not found: $f")
	end

	# Define and train an adaline for denoising
	printstyled("Training started...\n"; color = :magenta)
	adaline, loss_history = train_an_adaline_for_denoising(
		noisy_audio,
		clean_audio,
		window_size = 5,
	)

	# Denoising through the trained adaline
	printstyled("Denoising audio...\n"; color = :magenta)
	denoise_audio(input_audio, adaline, output_audio)
	printstyled("Denoised audio saved to $output_audio\n"; color = :light_green)

	# Optional: Plot training loss
	plot(loss_history, title = "Training Loss", label = "Error", lw = 2)
	lossimglog = "./log/loss_plot_old.png"
	savefig(lossimglog)
	printstyled("An image of training loss saved to $lossimglog\n"; color = :green)
	display(adaline)
end

main()


