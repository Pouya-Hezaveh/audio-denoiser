using Plots
using Random
using LinearAlgebra

mutable struct NeuralNetwork
	bias::Float64
	weights::Matrix{Float64}
	learning_rate::Float64
	epochs::Int
	loss_history::Vector{Float64}

	function NeuralNetwork(input_size::Union{Int, Tuple}; learning_rate = 0.001, epochs = 100)
		bias = randn()
		weights = randn(input_size)
		new(bias, weights, learning_rate, epochs, Float64[])
	end
end

function sigmoid(x::Float64)
	return 1.0 / (1.0 + exp(-x))
end

function sigmoid_derivative(x::Float64)
	return x * (1 - x)
end

function forward_propagation(network::NeuralNetwork, x::Vector{Float64})
	z = dot(network.weights, x) + network.bias
	return sigmoid(z)
end

function back_propagation(network::NeuralNetwork, x::Vector{Float64}, target::Float64, output::Float64)
	error = target - output
	d_output = error * sigmoid_derivative(output)

	# Update weights and bias
	network.weights .+= network.learning_rate * d_output * x
	network.bias += network.learning_rate * d_output
end

function train!(network::NeuralNetwork, X::Matrix{Float64}, y::Vector{Float64})
	for epoch in 1:network.epochs
		total_loss = 0.0
		for i in 1:size(X, 1)
			xi = X[i, :]
			target = y[i]

			# Forward
			output = forward_propagation(network, xi)

			total_loss += (target - output)^2

			# Backward
			back_propagation(network, xi, target, output)
		end

		# Store average loss for this epoch
		push!(network.loss_history, total_loss / size(X, 1))
	end
end

function main()
	X = [0 0; 0 1; 1 0; 1 1]
	y = [0, 1, 1, 0]

	nn = NeuralNetwork(2; learning_rate = 0.1, epochs = 10000)

	train!(nn, X, y)

	for i in 1:size(X, 1)
		output = forward_propagation(nn, X[i, :])
		println("Input: $(X[i, :]), Predicted Output: $output")
	end

	plot(nn.loss_history, title = "Training Loss", xlabel = "Epoch", ylabel = "Loss", lw = 2)
end

main()
