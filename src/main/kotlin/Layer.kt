import kotlin.math.sqrt
import kotlin.random.Random

class Layer(
    val size: Int,
    val previousSize: Int,
    val activation: ACTIVATION
    )
{
    var input = Array(size) { 0.0 }
    val output = Array(size) { 0.0 }

    val bias = Array(size) { 0.0 }

    val weight = Array(size) {
        Array(previousSize) { java.util.Random().nextGaussian(0.0, sqrt(2.0/(size+previousSize))) }
    }

    val delta = Array(size) { 0.0 }

    fun passThrough(input: Array<Double>): Array<Double> {
        for (neuronIndex in output.indices) {
            val sum = input.foldIndexed(0.0) { i, curr, elem -> curr + elem * weight[neuronIndex][i] } + bias[neuronIndex]
            this.input[neuronIndex] = sum
            this.output[neuronIndex] = activation.function(sum)
        }
        return output
    }

    fun calculateDeltas(weightedDeltaSum: Array<Double>) {
        weightedDeltaSum.forEachIndexed { index, delta ->
            this.delta[index] = delta * activation.derivative(input[index])
        }
    }

    fun getWeightedDeltaSum(): Array<Double> {
        val weightedDeltaSum = Array(previousSize) { prevLayerIndex ->
            (0 until size).fold(0.0) {  curr, currLayerIndex ->
                curr + weight[currLayerIndex][prevLayerIndex] * delta[currLayerIndex]
            }
        }
        return weightedDeltaSum
    }

    fun learn(prevLayerOutput: Array<Double>, learningRate: Double) {
        // adjust biases
        for (index in 0 until size) {
            val deltaBias = -delta[index] * learningRate
            bias[index] = (bias[index] + deltaBias).coerceIn(-.8, .8)
        }

        // adjust weights
        for (currentNeuronIndex in 0 until size) {
            for (prevNeuronIndex in 0 until previousSize) {
                val deltaWeight = - delta[currentNeuronIndex] * prevLayerOutput[prevNeuronIndex] * learningRate
                weight[currentNeuronIndex][prevNeuronIndex] += deltaWeight
            }
        }
    }

}

enum class ACTIVATION (
    val function: (Double) -> Double,
    val derivative: (Double) -> Double
) {
    SIGMOID(
        { x -> 1.0/(1+ kotlin.math.exp(-x)) },
        { x -> (1.0/(1+ kotlin.math.exp(-x))) * (1-(1.0/(1+ kotlin.math.exp(-x)))) }
    ),
    RELU(
        { x -> kotlin.math.max(0.0, x) },
        { x -> if (x < 0) 0.0 else 1.0 }
    ),
    LEAKYRELU (
        { x -> if (x < 0) 0.01*x else x },
        { x -> if (x < 0) 0.01 else 1.0 }
    )
}