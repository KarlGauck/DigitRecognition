import org.openrndr.draw.loadImage
import kotlin.random.Random

object DataManager {

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

    fun getRandomDataset(size: Int): MutableList<Array<Double>> {
        val choosenImages = mutableListOf<Pair<Int, Int>>()
        val dataset = mutableListOf<Array<Double>>()
        for (i in 0 until size) {
            var foundSet = false
            while (!foundSet) {
                val digit = Random.nextInt(10)
                val index = Random.nextInt(10000)
                val pair = Pair(digit, index)
                if (choosenImages.contains(pair))
                    continue
                choosenImages.add(pair)
                val data = readImage(digit, index)
                dataset.add(data)
                foundSet = true
            }
        }
        return dataset
    }

}