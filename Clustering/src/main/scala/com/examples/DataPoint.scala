package com.examples

class DataPoint(x : Float, y : Float) {
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
		new DataPoint(Math.random().floatValue(), Math.random().floatValue())
	}
}