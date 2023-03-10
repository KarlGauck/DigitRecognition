class Network(
    val layerSizes: Array<Int>
) {

    val layer = Array(layerSizes.size-1) {
        Layer(layerSizes[it+1], layerSizes[it], ACTIVATION.SIGMOID)
    }

    val learningRate = 10.0

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
        println()
        println()
        println("output: ")
        output.display()
        println("desiredOutput: ")
        desiredOutput.display()
        println("error:")
        val deltaerror = output.mapIndexed {index, it -> 2*(it - desiredOutput[index]) }.toTypedArray()
        deltaerror.display()

        println()
        println(" ===== Distribute Deltas ===== ")
        println("Layer | -----")
        for (layerIndex in layer.indices.reversed()) {
            val layer = layer[layerIndex]
            if (layerIndex == this.layer.size-1)
                layer.calculateDeltas(deltaerror)
            else
                layer.calculateDeltas(this.layer[layerIndex+1].getWeightedDeltaSum())
            print(" $layerIndex    | ")
            layer.delta.display()
        }

        layer.forEachIndexed { index, layer ->
            if (index == 0)
                layer.learn(input, learningRate)
            else layer.learn(this.layer[index-1].output, learningRate)
        }
    }

}