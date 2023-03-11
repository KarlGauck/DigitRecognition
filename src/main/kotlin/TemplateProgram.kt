import org.openrndr.application
import org.openrndr.draw.loadImage
import kotlin.math.sin
import kotlin.random.Random


fun main() = application {
    configure {
        width = 800
        height = 800
    }

    program {
        /*
        val w1 = 0.2
        val w2 = 0.3
        val b = 0.3

        val netorig = Net(2, 1)
        val netnew = Network(2, 1)

        netorig.weights[0][0][0] = w1
        netorig.weights[0][1][0] = w2
        netorig.biases[1][0] = b

        netnew.layer[0].weight[0][0] = w1
        netnew.layer[0].weight[0][1] = w2
        netnew.layer[0].bias[0] = b

        val i1 = 0.3
        val i2 = 0.4

        netorig.learn(arrayOf(i1, i1), arrayOf(i1+i2))
        netnew.learn(arrayOf(i1, i1), arrayOf(i1+i2))

        println("old: ")
        netorig.displayWeights()

        println("new: ")
        netnew.displayWeights()
        */

        /*
        val new = Network(2, 30, 1)
        val old = Net(2, 30, 1)

        val sampleSize = 200000
        for (sample in 0 until sampleSize) {
            val i1 = Random.nextDouble(.5)
            val i2 = Random.nextDouble(.5)
            old.learn(arrayOf(i1, i2), arrayOf(sin(i1+i2)))
            new.learn(arrayOf(i1, i2), arrayOf(sin(i1+i2)))
        }

        val testSize = 10
        for (test in 0 until testSize) {
            val i1 = Random.nextDouble(.5)
            val i2 = Random.nextDouble(.5)
            val oold = old.feedForward(arrayOf(i1, i2))
            val desiredOutput = sin(i1+i2)
            println("old: ${i1.format(3)} + ${i2.format(3)} = ${desiredOutput.format(3)}    network: ${oold[0].format(3)} error: ${(oold[0]-desiredOutput).format(3)}")
        }

        for (test in 0 until testSize) {
            val i1 = Random.nextDouble(.5)
            val i2 = Random.nextDouble(.5)
            val onew = new.feedForward(arrayOf(i1, i2))
            val desiredOutput = sin(i1+i2)
            println("new: ${i1.format(3)} + ${i2.format(3)} = ${desiredOutput.format(3)}    network: ${onew[0].format(3)} error: ${(onew[0]-desiredOutput).format(3)}")
        }

        new.displayWeights()
        new.displayBiases()

         */

        val network = Net(28*28, 100, 10)

        val trainingSteps = 1000
        for (step in 0 until trainingSteps) {
            for (digit in 0 until 10) {
                val data = readImage(digit, step)
                val desiredOutput = Array(10) { if (it == digit) 1.0 else 0.0 }
                network.learn(data, desiredOutput)
            }
        }

        val testSteps = 10
        for (digit in 0 until 10) {
            println("Digit: $digit")
            for (step in 0 until testSteps) {
                val data = readImage(digit, step)
                network.feedForward(data).display()
            }
        }
   }
}

fun readAndTrainBatched(batchCount: Int, batchSize: Int, network: Net) {
    for (batchIndex in 0 until batchCount) {
        val batch = mutableListOf<Pair<Array<Double>, Array<Double>>>()
        for (dataIndex in 0 until batchSize) {
            val index = batchIndex*batchSize + dataIndex
            for (digit in 0 until 10) {
                val data = readImage(digit, index)
                val desiredOutput = Array(10) { if (it == digit) 1.0 else 0.0 }
                batch.add(Pair(data, desiredOutput))
                //desiredOutput.display()
            }
        }
        network.batchLearning(batch.toTypedArray())
        println("Batch $batchIndex learned")
    }
}

fun readImage(digit: Int, index: Int): Array<Double> {
    val image = loadImage("./data/images/$digit/$digit/$index.png")
    val shadow = image.shadow
    shadow.download()
    val list = mutableListOf<Double>()
    for (y in 0 until image.effectiveWidth) {
        for (x in 0 until image.effectiveHeight) {
            list.add(shadow[x, y].alpha)
        }
    }
    shadow.destroy()
    return list.toTypedArray()
}

fun Double.format(digits: Int) = "%.${digits}f".format(this)

fun Array<Double>.display() {
    print("[")
    for (i in 0 until this.size)
        print("${this[i].format(3)}${if (i != this.size-1) " ; " else ""}")
    println("]")
}