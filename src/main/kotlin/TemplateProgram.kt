import org.openrndr.MouseEvent
import org.openrndr.application
import org.openrndr.color.ColorRGBa
import org.openrndr.color.mix
import org.openrndr.draw.Drawer
import org.openrndr.math.Vector2
import java.util.Random

var windowWidth: Int = 0
var windowHeight: Int = 0

var manager = NetworkManager()

fun main() = application {
    configure {
        width = 1600
        height = 850
    }

    program {
        manager.trainNetwork()

        windowWidth = width
        windowHeight = height
        mouse.dragged.listen {
            if (state == State.DRAWING)
                draw(it)
        }

        extend {
            logic()
            drawer.display()
        }
    }
}


var state = State.DRAWING
var data: Array<Double> = Array(28*28) {0.0}
var rotationData = mutableListOf<Array<Double>>()
var rotationIndex = 0

var image = Array(28) {
    Array(28) {
        0.0
    }
}

var lastRotation: Long = 0

fun logic() {
    when (state) {
        State.ROTATE -> {

            val rotationSize = 1000
            if (rotationData.size != rotationSize) {
                rotationData = DataManager.getRandomDataset(rotationSize)
            }
            val time = System.currentTimeMillis()
            if (time - lastRotation > 50){
                lastRotation = time
                rotationIndex = (rotationIndex+1)%rotationSize
                data = rotationData[rotationIndex]
            }

        }
        State.DRAWING -> {
            for (y in image[0].indices) {
                for (x in image.indices) {
                    val index = y*28+x
                    data[index] = image[x][y]
                }
            }
        }
    }
}

fun Drawer.display() {
    val cellSize = height/28.0
    val xOffset = (width-height)/2.0

    fill = ColorRGBa.RED
    rectangle(xOffset-4, 0.0, height.toDouble()+8, height.toDouble())
    for (x in 0 until 28) {
        for (y in 0 until 28) {
            val index = y*28+x
            val brightness = data[index]

            fill = mix(ColorRGBa.BLACK, ColorRGBa.WHITE, brightness)
            rectangle(x*cellSize+xOffset, y*cellSize, cellSize, cellSize)
        }
    }
}

fun draw(event: MouseEvent) {
    val cellSize = windowHeight/28.0
    val xOffset = (windowWidth-windowHeight)/2.0
    val pos = (event.position - Vector2(xOffset, 0.0))/cellSize
    val x = pos.x.toInt()
    val y = pos.y.toInt()
    for (dx in 0..1) {
        for (dy in 0..1) {
            if (x+dx !in image.indices || y+dy !in image[x+dx].indices)
                continue
            val brightness = if (dx==0 && dy==0) 1.0 else Random().nextGaussian(.8, .2)
            image[x+dx][y+dy] = brightness
        }
    }
}

enum class State {
    DRAWING,
    ROTATE
}