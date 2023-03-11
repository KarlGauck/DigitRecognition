import org.openrndr.draw.loadImage

class NetworkManager {


    fun trainNetwork() {
        val network = Network(28*28, 100, 10)

        val trainingSteps = 1000
        for (step in 0 until trainingSteps) {
            for (digit in 0 until 10) {
                val data = DataManager.readImage(digit, step)
                val desiredOutput = Array(10) { if (it == digit) 1.0 else 0.0 }
                network.learn(data, desiredOutput)
            }
        }

        val testSteps = 10
        for (digit in 0 until 10) {
            println("Digit: $digit")
            for (step in 0 until testSteps) {
                val data = DataManager.readImage(digit, step)
                network.feedForward(data).display()
            }
        }
    }



}