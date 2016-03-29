package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.hadoop.fs._
import org.rogach.scallop._
import java.lang.Float

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
	    FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)
	    
	    val textFile = sc.textFile(args.input())
	    				.map(line => {
	    					val arr = line.split(",")
	    					val x = Float.valueOf(arr(0).trim())
	    					val y = Float.valueOf(arr(1).trim())
	    					new DataPoint(x,y)
	    				})
	    				
	    val dataPoints = textFile.toArray() 				

		val centroids = Array.fill(7) { DataPoint.random }

		val resultCentroids = Kmeans(dataPoints, centroids, 0.0001f)
		log.info("***** resultCentroids size: " + resultCentroids.size)
		val gg = sc.parallelize(resultCentroids)
		
		resultCentroids.foreach(f => log.info("" + f))
		
		// Group the points according to their distance to centroid
		val cluster = dataPoints.groupBy { x =>
			{
				resultCentroids.reduceLeft(
					(a, b) => if (x.EuclideanDistance(a) < x.EuclideanDistance(b)) {
						a
					}
					else {
						b
					}
				)
			}
		}
		
		log.info("***** Cluster size: " + cluster.size)
		
		cluster.foreach(f => log.info("" + f._1))
				
		gg.saveAsTextFile(args.output())
	}

	def Kmeans(points : Seq[ DataPoint ], centroids : Seq[ DataPoint ], delta : Float) : Seq[ DataPoint ] = {

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
