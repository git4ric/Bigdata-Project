package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.rogach.scallop._

class Conf(args : Seq[ String ]) extends ScallopConf(args) {
	mainOptions = Seq(input, output)
	val input = opt[ String ](descr = "input path", required = true)
	val output = opt[ String ](descr = "output path", required = true)
}

object KMeans {
	val log = Logger.getLogger(getClass().getName())

	def main(argv : Array[ String ]) {

		var logger = Logger.getLogger(this.getClass())

		val jobName = "KMeans"

		val conf = new SparkConf().setAppName(jobName)
		val sc = new SparkContext(conf)
		val args = new Conf(argv)
		log.info("****** ~~~~~~ Input: " + args.input())
	    log.info("****** ~~~~~~ Output: " + args.output())
	    
	    val textFile = sc.textFile(args.input())
	    				.map(line => {
	    					val arr = line.split("\t")
	    					val x = arr(0).trim().toDouble
	    					val y = arr(1).trim().toDouble
	    					new DataPoint(x,y)
	    				})
	    				
	    val dataPoints = textFile.toArray() 				

		val centroids = Array.fill(7) { DataPoint.random }

		val resultCentroids = Kmeans(dataPoints, centroids, 0.2)
		println(resultCentroids)
	}

	def Kmeans(points : Seq[ DataPoint ], centroids : Seq[ DataPoint ], delta : Double) : Seq[ DataPoint ] = {

		// Group the points according to their distance to centroid
		val cluster = points.groupBy { x =>
			{
				centroids.reduceLeft(
					(a, b) => if (x.EuclideanDistance(a) < x.EuclideanDistance(b)) {
						a
					}
					else {
						b
					}
				)
			}
		}
		
		// Find new centroids but finding mean of all points in cluster
		val newCentroids = centroids.map(centroid => {
			
			val clusterPoints = cluster.getOrElse(centroid, List())			
			val clusterSize = clusterPoints.length
			
			if(clusterSize > 0)
			{
				clusterPoints.reduceLeft(_ + _) / clusterSize
			}
			else
			{
				centroid
			}
		})
		
		// Find out the change in centroids
		val centroidChange = (cluster.zip(newCentroids)).map(f => f._1._1.EuclideanDistance(f._2))
		
		if(centroidChange.exists(_ < delta))
		{
			newCentroids
		}
		else
		{
			Kmeans(points, newCentroids, delta)
		}
	}
}



