import org.openrndr.application
import org.openrndr.draw.loadImage
import kotlin.random.Random

val network = Network(arrayOf(2, 1))

fun main() = application {
    configure {
        width = 800
        height = 800
    }

    program {
        val sampleSize = 1000
        for (sample in 0 until sampleSize) {
            val i1 = Random.nextInt(2).toDouble()
            val i2 = Random.nextInt(2).toDouble()
            val o = if (i1==1.0 || i2==1.0) 1.0 else 0.0
            network.simpleLearning(arrayOf(i1, i2), arrayOf(o))
        }

        val tests = 10
        for (test in 0 until tests) {
            val i1 = Random.nextInt(2).toDouble()
            val i2 = Random.nextInt(2).toDouble()
            val o = if (i1==1.0 || i2==1.0) 1.0 else 0.0
            val res = network.feedForward(arrayOf(i1, i2))
            println("$i1 || $i2 = ${res[0]}")
        }

        /*
        val sampleSize = 1000
        for (sample in 0 until sampleSize) {
            network.displayWeights()
            network.displayBiases()
            println("")
            println("=============================")
            println("Sample $sample")
            val i1 = Random.nextDouble(.5)
            val i2 = Random.nextDouble(.5)
            network.simpleLearning(arrayOf(i1, i2), arrayOf(i1+i2))
        }

        val testSize = 10
        for (test in 0 until testSize) {
            val i1 = Random.nextDouble(.5)
            val i2 = Random.nextDouble(.5)
            val output = network.feedForward(arrayOf(i1, i2))
            val desiredOutput = i1+i2
            println("${i1.format(3)} + ${i2.format(3)} = ${desiredOutput.format(3)}    network: ${output[0].format(3)} error: ${(output[0]-desiredOutput).format(3)}")
        }
         */

        /*
        readAndTrain(0, 10)

        val testSamplesPerDigit = 10
        for (digit in 0 until 10) {
            println("-- -- -- Digit: $digit -- -- --")
            for (sample in 0 until testSamplesPerDigit) {
                val data = readImage(digit, sample)
                val output = network.feedForward(data)

                output.display()
            }
        }

         */
   }
}

fun readAndTrain(batchCount: Int, batchSize: Int) {
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