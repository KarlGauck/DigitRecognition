import kotlinx.coroutines.currentCoroutineContext
import kotlin.math.exp
import kotlin.math.sqrt
import kotlin.random.Random

class Network(val layerSizes: Array<Int>) {

    var weights: Array<Array<Array<Double>>>

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
                    Random.nextDouble()/layerSizes[layerI]
                }
            }
        }
    }

    var biases = Array(layerSizes.size) {
        Array(layerSizes[it]) {
            0.0
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

    private val delta = Array(layerSizes.size) {
        Array(layerSizes[it]) {
            0.0
        }
    }

    private var learningRate = 0.1

    fun feedForward(input: Array<Double>): Array<Double> {
        outputs[0] = input
        inputs[0] = input
        for (layerIndex in 1 until layerSizes.size) {
            val layerSize = layerSizes[layerIndex]
            for (currentNeuronIndex in 0 until layerSize) {
                var sum = 0.0
                for (prevNeuronIndex in 0 until layerSizes[layerIndex-1]) {
                    val weight = weights[layerIndex-1][prevNeuronIndex][currentNeuronIndex]
                    val signal = weight * outputs[layerIndex-1][prevNeuronIndex]
                    sum += signal
                }
                sum += biases[layerIndex][currentNeuronIndex]
                inputs[layerIndex][currentNeuronIndex] = sum
                outputs[layerIndex][currentNeuronIndex] = activation(sum)
            }
        }
        return outputs.last()
    }

    private fun propagateBack(input: Array<Double>, desiredOutput: Array<Double>): Pair<Array<Array<Array<Double>>>, Array<Array<Double>>> {
        feedForward(input)

        for (a in delta.indices) {
            for (b in delta[a].indices) {
                delta[a][b] = 0.0
            }
        }

        val newWeights = weights.clone()
        val newBiases = biases.clone()

        for (layerIndex in layerSizes.size-1 downTo 1) {
            println("- - - Layer: $layerIndex - - -")
            for (currentNeuronIndex in 0 until layerSizes[layerIndex]) {
                if (layerIndex == layerSizes.size-1) {
                    val error = (outputs[layerIndex][currentNeuronIndex] - desiredOutput[currentNeuronIndex])
                    println("error: $error")
                    println("delta: ${delta}")
                    delta[layerIndex][currentNeuronIndex] = activationDerivative(inputs[layerIndex][currentNeuronIndex]) * (outputs[layerIndex][currentNeuronIndex] - desiredOutput[currentNeuronIndex])
                }
                else {
                    println(delta[layerIndex][currentNeuronIndex])
                    delta[layerIndex][currentNeuronIndex] *= activationDerivative(inputs[layerIndex][currentNeuronIndex])
                }

                newBiases[layerIndex][currentNeuronIndex] -= delta[layerIndex][currentNeuronIndex]*learningRate
                for (previousNeuronIndex in 0 until  layerSizes[layerIndex-1]) {
                    val deltaWeight = -delta[layerIndex][currentNeuronIndex]*outputs[layerIndex-1][previousNeuronIndex]*learningRate
                    println(deltaWeight)
                    newWeights[layerIndex-1][previousNeuronIndex][currentNeuronIndex] += deltaWeight

                    delta[layerIndex-1][previousNeuronIndex] += weights[layerIndex-1][previousNeuronIndex][currentNeuronIndex]*delta[layerIndex][currentNeuronIndex]
                }
            }
        }

        return Pair(newWeights, newBiases)
    }

    fun simpleLearning(input: Array<Double>, output: Array<Double>) {
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

    private fun activation(x: Double): Double = 1.0/(1+exp(-x))

    private fun activationDerivative(x: Double) = activation(x) * (1-activation(x))

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

}