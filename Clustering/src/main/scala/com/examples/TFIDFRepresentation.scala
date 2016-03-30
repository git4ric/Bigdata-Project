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
		dotProd(vec1, vec2).toFloat / (norm(vec1) * norm(vec2))
	}

	def cosineDistance(vec1 : Map[ Int, Double ], vec2 : Map[ Int, Double ]) : Double = {
		val result = 1 - (Math.acos(cosineSimilarity(vec1, vec2)) / 3.14)
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

		val hconf = new Configuration
		hconf.set("textinputformat.record.delimiter", "#Article:")

		val dataset = sc.newAPIHadoopFile(args.input(), classOf[ TextInputFormat ], classOf[ LongWritable ], classOf[ Text ], hconf)
			.map(x => x._2.toString())
			.filter(x => x.isEmpty() == false)
			.map(x => x.replaceAll("#Type: regular article", "")
				.replaceAll("\\W", " ")
				.replaceAll("\\s\\s+", " ")
				.split(" ").toSeq.drop(1))

		val hashingTF = new HashingTF()
		val tf : RDD[ Vector ] = hashingTF.transform(dataset)
		tf.cache()
		val idf = new IDF().fit(tf)
		val tfidf : RDD[ Vector ] = idf.transform(tf)

		val gg = tfidf.map(x => x.toSparse)

		val articles = gg.map(x => (x.indices zip x.values).toMap)

		println("Dataset: ")
		articles.foreach(println)

		var centroids = articles.takeSample(false, args.clusters().toInt, 20)

		println("Start centroids")
		println(centroids.deep.mkString("\n"))

		var iteration = 0

		while (iteration < args.iterations().toInt) {

			// Get the closest centroid to each article
			// and map them as centroid -> (article,1)
			// Merge the articles and sum their occurrence within each centroid cluster to create a new centroid
			val clusters = articles.map(article => (closestCentroid(article, centroids), (article, 1)))
				.reduceByKeyLocally({
					case ((articleA, occurA), (articleB, occurB)) => (mergeMap(articleA, articleB), occurA + occurB)
				})

			// Divide each value of new centroid by cluster size to get mean	
			val average = clusters.map({
				case (centroid, (newCentroid, clusterSize)) =>
					(centroid, newCentroid.map(x => (x._1, x._2 / clusterSize)))
			})

			// Update centroid	
			centroids = centroids.map(oldCentroid => {
				average.get(oldCentroid) match {
					case Some(newCentroid) => newCentroid
					case None => oldCentroid
				}
			})

			iteration = iteration + 1

		}

		println(centroids.deep.mkString("\n"))

	}
}


