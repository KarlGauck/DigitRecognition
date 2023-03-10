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
        Array(previousSize) { Random.nextDouble(.1, .221) }
    }

    val delta = Array(size) { 0.0 }

    fun passThrough(input: Array<Double>): Array<Double> {
        this.input = input
        for (neuronIndex in output.indices) {
            val sum = input.foldIndexed(0.0) { i, curr, elem -> curr + elem * weight[neuronIndex][i] } / input.size + bias[neuronIndex]
            val output = activation.function(sum)
            this.output[neuronIndex] = output
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
            val deltaBias = - delta[index] * learningRate
            println("db $index: $deltaBias")
            bias[index] -= deltaBias
        }
        println()

        // adjust weights
        for (currentNeuronIndex in 0 until size) {
            for (prevNeuronIndex in 0 until previousSize) {
                val deltaWeight = - delta[currentNeuronIndex] * prevLayerOutput[prevNeuronIndex] * learningRate
                weight[currentNeuronIndex][prevNeuronIndex] += deltaWeight
                println("dw $currentNeuronIndex $prevNeuronIndex: $deltaWeight")
            }
        }
    }

}

enum class ACTIVATION (
    val function: (Double) -> Double,
    val derivative: (Double) -> Double
) {
    SIGMOID(
        { x -> 1.0/(1+ kotlin.math.exp(-x))},
        {x -> (1.0/(1+ kotlin.math.exp(-x))) * (1-(1.0/(1+ kotlin.math.exp(-x))))}
    )
}