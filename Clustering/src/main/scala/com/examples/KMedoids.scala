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

object KMedoids {
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

	def medoidDistance(vec1 : Map[ Int, Double ]) : Double = {
		0.0
	}

	def main(argv : Array[ String ]) {

		var logger = Logger.getLogger(this.getClass())

		val jobName = "KMedoids"

		val conf = new SparkConf().setAppName(jobName)
		val sc = new SparkContext(conf)
		val args = new Conf(argv)
		log.info("****** ~~~~~~ Input: " + args.input())
		log.info("****** ~~~~~~ Output: " + args.output())
		log.info("****** ~~~~~~ No. of Clusters: " + args.clusters())
		log.info("****** ~~~~~~ No. of iterations: " + args.iterations())
		FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)

		val hconf = new Configuration
		hconf.set("textinputformat.record.delimiter", "#Article:")

		val dataset = sc.newAPIHadoopFile(args.input(), classOf[ TextInputFormat ], classOf[ LongWritable ], classOf[ Text ], hconf)
			.map(x => x._2.toString())
			.filter(x => x.isEmpty() == false)
			.map(x => x.replaceAll("#Type: regular article", "")
				.replaceAll("[^a-zA-Z]", " ")
				.replaceAll("\\s\\s+", " ")
				.split(" ").toSeq.drop(1))

		val hashingTF = new HashingTF()
		val tf : RDD[ Vector ] = hashingTF.transform(dataset)
		tf.cache()
		val idf = new IDF(minDocFreq = 1).fit(tf)
		val tfidf : RDD[ Vector ] = idf.transform(tf)

		val gg = tfidf.map(x => x.toSparse)

		val articles = gg.map(x => (x.indices zip x.values).toMap)

		//		println("Data: ")
		//		articles.foreach(println)

		var medoids = articles.takeSample(false, args.clusters().toInt)

		println("Start medoids")
		println(medoids.deep.mkString("\n"))

		val test = articles.collect()

		//		for (i <- 0 until test.length) {			
		//			for(j <- 0 until medoids.length){
		//				val distance = cosineDistance(test(i), medoids(j))
		//				println("Distance between " + i.toString() + " and " + j.toString() + " is: " + distance.toString())
		//			}
		//			
		//		}

		var iteration = 0

		while (iteration < args.iterations().toInt) {

			// Get the closest medoids to each article
			// and map them as medoids -> (article)
			val clusters = articles.map(article => (closestCentroid(article, medoids), article)).groupByKey()

			//			println("Cluster count: " + clusters.count().toString())
			//			clusters.keys.foreach(println)

			val newMedoids = clusters.map(f => {

				val m = f._1
				var best = Double.PositiveInfinity
				var bestMedoid = f._1

				val temp = f._2

				f._2.foreach(p => {

					val o = p
					var distance = 0.0
					temp.foreach(x => {
						distance = distance + cosineDistance(o, x)
					})

					distance = distance + cosineDistance(o, m)
					distance = distance / temp.size

					if (distance < best) {
						best = distance
						bestMedoid = o
					}
				})
				(bestMedoid)
			})

			//			println("New Medoids size: " + newMedoids.count().toString())

			medoids = newMedoids.collect().clone()

			iteration = iteration + 1
		}

		val printThis = averageDistanceBetweenCentroids(medoids)

		println(printThis.mkString("\n"))

		println("medoids")
		println(medoids.deep.mkString("\n"))
	}
}



