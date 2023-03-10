import kotlin.math.exp
import kotlin.math.tanh
import kotlin.random.Random

class Net(val layerSizes: Array<Int>) {

    var weights: Array<Array<Array<Double>>>

    var deltaWeights = Array(layerSizes.size-1) { layer ->
        Array(layerSizes[layer]) {
            Array(layerSizes[layer+1]) {
                0.0
            }
        }
    }

    init {
        layerSizes.forEach(::println)
        weights = Array(layerSizes.size-1) {
            Array(0) {
                Array(0) {
                    0.0
                }
            }
        }
        for (layerI in weights.indices) {
            weights[layerI] = Array(layerSizes[layerI]) {
                Array(layerSizes[layerI+1]) {
                    Random.nextDouble(.5, 2.0)
                }
            }
        }
    }

    var biases = Array(layerSizes.size) {
        Array(layerSizes[it]) {
            Random.nextDouble(.3, .9)
        }
    }

    val outputs = Array(layerSizes.size) {
        Array(layerSizes[it]) {
            0.0
        }
    }

    val inputs = Array(layerSizes.size) {
        Array(layerSizes[it]) {
            0.0
        }
    }

    private var learningRate = 5.0

    private val inertia = .8

    private var activation = ACTIVATION.SIGMOID

    private val printoutput = false

    fun feedForward(input: Array<Double>): Array<Double> {
        if (printoutput) println("-- -- -- FEED FORWARD -- -- --")
        outputs[0] = input
        inputs[0] = input
        for (layerIndex in 1 until layerSizes.size) {
            if (printoutput) println("  current layer: $layerIndex")
            val layerSize = layerSizes[layerIndex]
            for (currentNeuronIndex in 0 until layerSize) {
                if (printoutput) println("    current neuron: $currentNeuronIndex")
                if (printoutput) println("      sum = 0")
                var sum = 0.0
                for (prevNeuronIndex in 0 until layerSizes[layerIndex-1]) {
                    if (printoutput) println("        previousNeuron: $prevNeuronIndex")
                    val weight = weights[layerIndex-1][prevNeuronIndex][currentNeuronIndex]
                    if (printoutput) println("          weight to neuron is $weight")
                    val output = outputs[layerIndex-1][prevNeuronIndex]
                    if (printoutput) println("          output of neuron was $output")
                    val signal = weight * outputs[layerIndex-1][prevNeuronIndex]
                    if (printoutput) println("          signal is $signal")
                    sum += signal
                    if (printoutput) println("      sum = $sum")
                }
                val bias = biases[layerIndex][currentNeuronIndex]
                if (printoutput) println("      bias is $bias")
                sum += biases[layerIndex][currentNeuronIndex]
                if (printoutput) println("      adding bias, sum = $sum")
                inputs[layerIndex][currentNeuronIndex] = sum
                outputs[layerIndex][currentNeuronIndex] = activation(sum)
                if (printoutput) println("      output now is ${activation(sum)}")
            }
        }
        return outputs.last()
    }

    private fun propagateBack(input: Array<Double>, desiredOutput: Array<Double>): Pair<Array<Array<Array<Double>>>, Array<Array<Double>>> {
        feedForward(input)

        if (printoutput) println("")
        if (printoutput) println("-- -- -- PROPAGATE BACK -- -- --")

        val delta = Array(layerSizes.size) {
            Array(layerSizes[it]) {
                0.0
            }
        }

        val newWeights = weights.clone()
        val newBiases = biases.clone()

        for (layerIndex in layerSizes.size-1 downTo 1) {
            if (printoutput) println("-- layer: $layerIndex --")
            for (currentNeuronIndex in 0 until layerSizes[layerIndex]) {
                if (printoutput) println(" - neuron: $currentNeuronIndex - ")
                if (printoutput) println("  - calculate delta")
                if (layerIndex == layerSizes.size-1) {
                    delta[layerIndex][currentNeuronIndex] = activationDerivative(inputs[layerIndex][currentNeuronIndex]) * (outputs[layerIndex][currentNeuronIndex] - desiredOutput[currentNeuronIndex])
                    if (printoutput) println("   we are in outputlayer")
                    val input = inputs[layerIndex][currentNeuronIndex]
                    if (printoutput) println("   input was $input")
                    val derivative = activationDerivative(input)
                    if (printoutput) println("   derivative at input is $derivative")
                    if (printoutput) println("    ***")
                    val output = outputs[layerIndex][currentNeuronIndex]
                    if (printoutput) println("   output was $output")
                    val y = desiredOutput[currentNeuronIndex]
                    if (printoutput) println("   desired output was $y")
                    val error = output-y
                    if (printoutput) println("   therfore, error derivative is $error")
                    val delta = delta[layerIndex][currentNeuronIndex]
                    if (printoutput) println("    ***")
                    if (printoutput) println("   delta now is $delta")
                }
                else {
                    if (printoutput) println("   we are not in outputlayer")
                    val deltasum = delta[layerIndex][currentNeuronIndex]
                    if (printoutput) println("   sum of input was $deltasum")
                    val input = inputs[layerIndex][currentNeuronIndex]
                    if (printoutput) println("   input was $input")
                    val derivative = activationDerivative(input)
                    if (printoutput) println("   derivative at input is $derivative")
                    if (printoutput) println("    ***")
                    val newdelta = deltasum * derivative
                    if (printoutput) println("   therefore newdelta is $newdelta")
                    delta[layerIndex][currentNeuronIndex] *= activationDerivative(inputs[layerIndex][currentNeuronIndex])
                }

                newBiases[layerIndex][currentNeuronIndex] -= delta[layerIndex][currentNeuronIndex]*learningRate

                if (printoutput) println("  - distribute delta")
                val deltaval = delta[layerIndex][currentNeuronIndex]
                if (printoutput) println("   delta: $deltaval")
                for (previousNeuronIndex in 0 until  layerSizes[layerIndex-1]) {
                    if (printoutput) println("   - prev Neuron: $previousNeuronIndex")
                    val output = outputs[layerIndex-1][previousNeuronIndex]
                    if (printoutput) println("     output of neuron: $output")
                    val deltaWeight = -delta[layerIndex][currentNeuronIndex]*outputs[layerIndex-1][previousNeuronIndex]*learningRate
                    if (printoutput) println("     deltaWeight = $deltaWeight")
                    deltaWeights[layerIndex-1][previousNeuronIndex][currentNeuronIndex] = deltaWeight
                    val oldWeight = weights[layerIndex-1][previousNeuronIndex][currentNeuronIndex]
                    if (printoutput) println("     old weight was $oldWeight")
                    newWeights[layerIndex-1][previousNeuronIndex][currentNeuronIndex] += deltaWeight
                    val newWeight = newWeights[layerIndex-1][previousNeuronIndex][currentNeuronIndex]
                    if (printoutput) println("     new weight is $newWeight")

                    if (printoutput) println("      ***")
                    // weight is adjusted, so that the sum of all weight*delta are equal to the original
                    val oldDelta = delta[layerIndex-1][previousNeuronIndex]
                    if (printoutput) println("     old delta was $oldDelta")
                    delta[layerIndex-1][previousNeuronIndex] += weights[layerIndex-1][previousNeuronIndex][currentNeuronIndex]*delta[layerIndex][currentNeuronIndex]
                    val newDelta = delta[layerIndex-1][previousNeuronIndex]
                    if (printoutput) println("     new delta is $newDelta")
                }
            }
        }

        return Pair(newWeights, newBiases)
    }

    fun learn(input: Array<Double>, output: Array<Double>) {
        val result = propagateBack(input, output)
        weights = result.first
        biases = result.second
    }

    fun batchLearning(batch: Array<Pair<Array<Double>, Array<Double>>>) {
        val weightSum = Array(weights.size) { a ->
            Array(weights[a].size) { b ->
                Array(weights[a][b].size) {
                    0.0
                }
            }
        }
        val biasSum = Array(biases.size) {
            Array(biases[it].size) {
                0.0
            }
        }

        for (data in batch) {
            val trainingResult = propagateBack(data.first, data.second)
            val newWeights = trainingResult.first
            val newBiases = trainingResult.second

            for (layerIndex in newWeights.indices) {
                for (firstNeuronIndex in newWeights[layerIndex].indices) {
                    for (secondNeuronIndex in newWeights[layerIndex][firstNeuronIndex].indices) {
                        weightSum[layerIndex][firstNeuronIndex][secondNeuronIndex] += newWeights[layerIndex][firstNeuronIndex][secondNeuronIndex]
                    }
                }
            }

            for (layerIndex in newBiases.indices) {
                for (neuronIndex in newBiases[layerIndex].indices) {
                    biasSum[layerIndex][neuronIndex] += newBiases[layerIndex][neuronIndex]
                }
            }
        }

        for (layerIndex in weightSum.indices) {
            for (firstNeuronIndex in weightSum[layerIndex].indices) {
                for (secondNeuronIndex in weightSum[layerIndex][firstNeuronIndex].indices) {
                    weights[layerIndex][firstNeuronIndex][secondNeuronIndex] = weightSum[layerIndex][firstNeuronIndex][secondNeuronIndex]/batch.size
                }
            }
        }

        for (layerIndex in biasSum.indices) {
            for (neuronIndex in biasSum[layerIndex].indices) {
                biases[layerIndex][neuronIndex] = biasSum[layerIndex][neuronIndex]/batch.size
            }
        }
    }

    private fun activation(x: Double): Double = when (activation) {
        ACTIVATION.TANH -> tanh(x)
        ACTIVATION.SIGMOID -> 1.0/(1+exp(-x))
    }

    private fun activationDerivative(x: Double) = when(activation) {
        ACTIVATION.TANH -> 1-activation(2*x)
        ACTIVATION.SIGMOID -> activation(x) * (1-activation(x))
    }

    fun displayWeights() {
        println("-- -- -- Weights -- -- --")
        for (layerIndex in weights.indices) {
            println("From layer $layerIndex to ${layerIndex+1}")
            for (prevNodeIndex in weights[layerIndex].indices) {
                for (thisNodeIndex in weights[layerIndex][prevNodeIndex].indices) {
                    println("    w from $prevNodeIndex to $thisNodeIndex: ${weights[layerIndex][prevNodeIndex][thisNodeIndex]}")
                }
            }
        }
    }

    fun displayBiases() {
        println("-- -- -- Biases -- -- --")
        for (layerIndex in biases.indices) {
            println("Layer ${layerIndex+1}")
            for (neuronIndex in biases[layerIndex].indices) {
                println("      Bias $neuronIndex: ${biases[layerIndex][neuronIndex]}")
            }
        }
    }

    fun displayDeltaWeights() {
        println("-- -- -- Delta Weights -- -- --")
        for (layerIndex in deltaWeights.indices) {
            println("From layer $layerIndex to ${layerIndex+1}")
            for (prevNodeIndex in deltaWeights[layerIndex].indices) {
                for (thisNodeIndex in deltaWeights[layerIndex][prevNodeIndex].indices) {
                    println("    w from $prevNodeIndex to $thisNodeIndex: ${deltaWeights[layerIndex][prevNodeIndex][thisNodeIndex]}")
                }
            }
        }
    }

    enum class ACTIVATION {
        SIGMOID,
        TANH
    }

}