package com.examples

@SerialVersionUID(100L)
@serializable case class DataPoint(x : Float, y : Float) {
	val myX = x
	val myY = y
	
	def +(that: DataPoint) : DataPoint = {
		new DataPoint(this.myX + that.myX, this.myY + that.myY)
	}
	
	def /(that: Float) : DataPoint = {
		new DataPoint(this.myX / that, this.myY/that)
	}
	
	def EuclideanDistance(other : DataPoint) : Float = {
		val result = Math.sqrt((this.myX - other.myX)*(this.myX - other.myX)
								+ (this.myY - other.myY)*(this.myY - other.myY))
		result.floatValue()					
	}
	
} 

object DataPoint {
	
	def random() = {
		val rnd = new scala.util.Random
		val range = 0 to 30
		new DataPoint(rnd.nextInt(range length).toFloat, rnd.nextInt(range length).toFloat)  
	}
}
