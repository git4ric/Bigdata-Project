package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger
import org.apache.hadoop.fs._
import org.rogach.scallop._
import java.lang.Float

import org.apache.hadoop.mapreduce.lib.input.TextInputFormat
import org.apache.hadoop.conf.Configuration
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.feature.IDF
import org.apache.hadoop.io.{LongWritable, Text}
import org.apache.spark.rdd.RDD

object TFIDFRepresentation {
	val log = Logger.getLogger(getClass().getName())

	def main(argv : Array[ String ]) {

		var logger = Logger.getLogger(this.getClass())

		val jobName = "TFIDFRepresentation"

		val conf = new SparkConf().setAppName(jobName)	
		val sc = new SparkContext(conf)
		val args = new Conf(argv)
		log.info("****** ~~~~~~ Input: " + args.input())
	    log.info("****** ~~~~~~ Output: " + args.output())
	    log.info("****** ~~~~~~ No. of Clusters: " + args.clusters())
	    FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)
	    
	    val hconf = new Configuration
	    hconf.set("textinputformat.record.delimiter", "#Article:")
	    
	    val dataset = sc.newAPIHadoopFile(args.input(), classOf[TextInputFormat], classOf[LongWritable], classOf[Text], hconf)
	    				.map(x => x._2.toString())
						.filter( x => x.isEmpty() == false)
	    				.map(x => x.replaceAll("#Type: regular article","")
	    						.replaceAll("\\W", " ")
								.replaceAll("\\s\\s+"," ")
								.split(" ").toSeq.drop(1))
	    				
	    val hashingTF = new HashingTF()
		val tf: RDD[Vector] = hashingTF.transform(dataset)
	    tf.cache()
		val idf = new IDF(minDocFreq = 2).fit(tf)
		val tfidf: RDD[Vector] = idf.transform(tf)
		
		val gg = tfidf.map(x => x.toSparse)
		
		val hh = gg.map(x => (x.indices zip x.values).toMap)

		val centroids = hh.takeSample(false, args.clusters().toInt, 20)
		
		val resultCentroids = Kmeans(hh,centroids,0.1,0)	
		
		println(resultCentroids.deep.mkString("\n"))
		
	}
	
	def dotProd(vec1: Map[Int,Double], vec2: Map[Int,Double]) : Double = {
		var sum = 0.0
		
		vec1.map(f => if(vec2.contains(f._1)){
			sum = sum + (f._2 * vec2.apply(f._1))			
		})
		sum
	}
	
	def norm(vec: Map[Int,Double]) : Double = {
		val dot = dotProd(vec,vec)
		val result = math.sqrt(dot)
		result
	}
	
	def cosineSimilarity(vec1: Map[Int,Double], vec2: Map[Int,Double]) : Double = {
		dotProd(vec1,vec2).toFloat / (norm(vec1) * norm(vec2)) 
	}
	
	def cosineDistance(vec1: Map[Int,Double], vec2: Map[Int,Double]) : Double = {
		val result = 1 - (Math.acos(cosineSimilarity(vec1, vec2)) / 3.14)
		result
	}
	
	def mergeMap(vec1: Map[Int,Double], vec2: Map[Int,Double]) : Map[Int, Double] = {
		val list = vec1.toList ++ vec2.toList
		val merged = list.groupBy ( _._1) .map { case (k,v) => k -> v.map(_._2).sum }
		merged
	}
	
	
	def Kmeans(points : RDD[Map[Int,Double]], centroids : Array[Map[Int,Double] ], delta: Double, iteration: Int) : Array[Map[Int,Double]] = {

		// Group the points according to their distance to centroid
		val cluster = points.groupBy { x =>
			{
				centroids.reduceLeft(
					(a, b) => if (cosineDistance(x,a) < cosineDistance(x,b)) {
						a
					}
					else {
						b
					}
				)
			}
		}
		
		// Find new centroids by finding mean of all points in cluster
		val newCentroids = centroids.map(centroid => {
			
			val clusterPoints = cluster.flatMap(f => f._2)		
			val clusterSize = clusterPoints.count()
			
			if(clusterSize > 0)
			{
				val newCentroid = clusterPoints.reduce((a,b) => mergeMap(a,b)).map(f => (f._1,f._2/clusterSize))
				newCentroid
			}
			else
			{
				centroid
			}
		})
		
		val centroidChange = (centroids zip newCentroids).map({ case (a, b) => cosineDistance(a, b) })
		iteration.set(iteration + 1)
		
		if(centroidChange.exists(_ < delta) || iteration > 5)
		{
			newCentroids
		}
		else
		{
			Kmeans(points, newCentroids, delta, iteration)
		}
	}
}

