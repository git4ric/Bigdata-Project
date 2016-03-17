package com.examples

class DataPoint(x : Double, y : Double) {
	val myX = x
	val myY = y
	
	def +(that: DataPoint) : DataPoint = {
		new DataPoint(this.myX + that.myX, this.myY + that.myY)
	}
	
	def /(that: Double) : DataPoint = {
		new DataPoint(this.myX / that, this.myY/that)
	}
	
	def EuclideanDistance(other : DataPoint) : Double = {
		val result = Math.sqrt((this.myX - other.myX)*(this.myX - other.myX)
								+ (this.myY - other.myY)*(this.myY - other.myY))
		result						
	}
}

object DataPoint {
	def random() = {
		new DataPoint(Math.random(), Math.random())
	}
}