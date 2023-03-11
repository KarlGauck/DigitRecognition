fun Double.format(digits: Int) = "%.${digits}f".format(this)

fun Array<Double>.display() {
    print("[")
    for (i in indices)
        print("${this[i].format(3)}${if (i != this.size-1) " ; " else ""}")
    println("]")
}