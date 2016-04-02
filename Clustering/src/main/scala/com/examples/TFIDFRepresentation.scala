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
import org.apache.hadoop.io.{ LongWritable, Text }
import org.apache.spark.rdd.RDD

object TFIDFRepresentation {
	val log = Logger.getLogger(getClass().getName())

	def closestCentroid(vec : Map[ Int, Double ], centroids : Array[ Map[ Int, Double ] ]) : Map[ Int, Double ] = {
		var distance = Double.PositiveInfinity
		var bestIndex = 0
		for (i <- 0 until centroids.length) {
			val tempDistance = cosineDistance(vec, centroids(i))
			if (tempDistance < distance) {
				distance = tempDistance
				bestIndex = i
			}
		}
		centroids(bestIndex)
	}

	def averageDistanceBetweenCentroids(centroids : Array[ Map[ Int, Double ] ]) : Array[ Double ] = {

		var result = new Array[ Double ](centroids.size)

		for (i <- 0 until centroids.length) {
			var distance_i = 0.0
			for (j <- 0 until centroids.length) {
				if (i != j) {
					distance_i = distance_i + cosineDistance(centroids(i), centroids(j))
				}
			}
			result.update(i, distance_i / centroids.size)
		}
		result
	}

	def dotProd(vec1 : Map[ Int, Double ], vec2 : Map[ Int, Double ]) : Double = {
		var sum = 0.0

		vec1.map(f => if (vec2.contains(f._1)) {
			sum = sum + (f._2 * vec2.apply(f._1))
		})
		sum
	}

	def norm(vec : Map[ Int, Double ]) : Double = {
		val dot = dotProd(vec, vec)
		val result = math.sqrt(dot)
		result
	}

	def cosineSimilarity(vec1 : Map[ Int, Double ], vec2 : Map[ Int, Double ]) : Double = {
		dotProd(vec1, vec2) / (norm(vec1) * norm(vec2))
	}

	def cosineDistance(vec1 : Map[ Int, Double ], vec2 : Map[ Int, Double ]) : Double = {
		val result = 1 - cosineSimilarity(vec1, vec2)
		result
	}

	def mergeMap(vec1 : Map[ Int, Double ], vec2 : Map[ Int, Double ]) : Map[ Int, Double ] = {
		val list = vec1.toList ++ vec2.toList
		val merged = list.groupBy(_._1).map { case (k, v) => k -> v.map(_._2).sum }
		merged
	}

	def main(argv : Array[ String ]) {

		var logger = Logger.getLogger(this.getClass())

		val jobName = "TFIDFRepresentation"

		val conf = new SparkConf().setAppName(jobName)
		val sc = new SparkContext(conf)
		val args = new Conf(argv)
		log.info("****** ~~~~~~ Input: " + args.input())
		log.info("****** ~~~~~~ Output: " + args.output())
		log.info("****** ~~~~~~ No. of Clusters: " + args.clusters())
		log.info("****** ~~~~~~ No. of iterations: " + args.iterations())
		FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)

//		val hconf = new Configuration
//		hconf.set("textinputformat.record.delimiter", ".I")

//		val dataset = sc.newAPIHadoopFile(args.input(), classOf[ TextInputFormat ], classOf[ LongWritable ], classOf[ Text ], hconf)
//			.map(x => x._2.toString())
//			.filter(x => x.isEmpty() == false)
//			.map(x => {
//				val y = x.split(".W")
//				(y(0).trim().toInt,y(1).trim().split(" ").toSeq)						
//			})
			
		val dataset = sc.textFile(args.input())
						.map(x => {
							val y = x.split("\\t")
							val z = y(0)
							val a = y(1).trim().split(" ").toSeq
							(z,a)
						})
				
//		println("Dataset: ")
//		dataset.foreach(println)

		val hashingTF = new HashingTF()
		
		val tf = dataset.map(x => (x._1,hashingTF.transform(x._2))).cache()
		val idf = new IDF(minDocFreq = 50).fit(tf.values)
		
		val tfidf = tf.map(x => (x._1,idf.transform(x._2))) 

		val gg = tfidf.map(x => (x._1,x._2.toSparse))

		val articles = gg.map(x => (x._1,(x._2.indices zip x._2.values).toMap)).cache()

//		println("Articles: ")
//		articles.foreach(println)

		var centroids = articles.takeSample(false, args.clusters().toInt).map(x => x._2)

//		println("Start centroids")
//		println(centroids.deep.mkString("\n"))

		var iteration = 0

		while (iteration < args.iterations().toInt) {

			// Get the closest centroid to each article
			// and map them as centroid -> (article,1)
			val clusters = articles.map(article => (closestCentroid(article._2, centroids), (article._2, 1)))

			// Merge the articles and sum their occurrence within each 
			// centroid cluster to create a new centroid
			val newCentroids = clusters.reduceByKeyLocally({
				case ((articleA, occurA), (articleB, occurB)) => (mergeMap(articleA, articleB), occurA + occurB)
			})

			// Divide each value of new centroid by cluster size to get mean	
			val average = newCentroids.map({
				case (centroid, (newCentroid, clusterSize)) =>
					(centroid, newCentroid.map(x => (x._1, x._2 / clusterSize)))
			})

			// Update centroid	
			var cent = centroids.map(oldCentroid => {
				average.get(oldCentroid) match {
					case Some(newCentroid) => newCentroid
					case None => oldCentroid
				}
			})
			
			// Compare cent and centroid to find convergence
			// val converge = (cent zip centroids).map{case (a,b) => cosineDistance(a,b)}

			iteration = iteration + 1

		}
		
		val clusters = articles.map(article => (closestCentroid(article._2, centroids), article._1)).groupByKey().map(x => (x._1,x._2.count(x => (x.isEmpty() == false))))
		clusters.coalesce(1, false).saveAsTextFile(args.output())
		val printThis = averageDistanceBetweenCentroids(centroids)
		println("***** ~~~~ Average Distance between Centroids: " + printThis.toString())
	}
}

