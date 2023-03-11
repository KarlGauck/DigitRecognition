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
        val network = Network(28*28, 100, 10)

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