class Network(
    vararg layerSizes: Int
) {

    val layer = Array(layerSizes.size-1) {
        Layer(layerSizes[it+1], layerSizes[it], ACTIVATION.SIGMOID)
    }

    val learningRate = 1.0

    fun feedForward(input: Array<Double>): Array<Double> {
        for (layerIndex in layer.indices) {
            val layer = this.layer[layerIndex]
            if (layerIndex == 0)
                layer.passThrough(input)
            else
                layer.passThrough(this.layer[layerIndex-1].output)
        }
        return layer.last().output
    }

    fun feedForward(input: Array<Double>, layerIndex: Int = 0): Array<Double> {
        return if (layerIndex != layer.size-1) feedForward(layer[layerIndex].passThrough(input), layerIndex+1) else layer[layerIndex].passThrough(input)
    }

    fun learn(input: Array<Double>, desiredOutput: Array<Double>) {
        val output = feedForward(input)
        val deltaerror = output.mapIndexed {index, it -> (it - desiredOutput[index]) }.toTypedArray()

        for (layerIndex in layer.indices.reversed()) {
            val layer = layer[layerIndex]
            if (layerIndex == this.layer.size-1)
                layer.calculateDeltas(deltaerror)
            else
                layer.calculateDeltas(this.layer[layerIndex+1].getWeightedDeltaSum())
        }

        layer.forEachIndexed { index, layer ->
            if (index == 0)
                layer.learn(input, learningRate)
            else layer.learn(this.layer[index-1].output, learningRate)
        }
    }

    fun displayWeights() {
        println("-- -- -- Weights -- -- --")
        for (layerIndex in layer.indices) {
            println("From layer $layerIndex to ${layerIndex+1}")
            for (prevNodeIndex in layer[layerIndex].weight[0].indices) {
                for (thisNodeIndex in layer[layerIndex].weight.indices) {
                    println("    w from $prevNodeIndex to $thisNodeIndex: ${layer[layerIndex].weight[thisNodeIndex][prevNodeIndex]}")
                }
            }
        }
    }

    fun displayBiases() {
        for (layer in layer) {
            layer.bias.display()
        }
    }

    fun displayDeltas() {
        for (layer in layer) {
            layer.delta.display()
        }
    }
}