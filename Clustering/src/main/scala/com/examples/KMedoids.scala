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

	def closestCentroid(vec : Map[ Int, Double ], centroids : Array[ Map[ Int, Double ] ]) : (Int, Map[ Int, Double ]) = {
		var distance = Double.PositiveInfinity
		var bestIndex = 0
		for (i <- 0 until centroids.length) {
			val tempDistance = cosineDistance(vec, centroids(i))
			if (tempDistance < distance) {
				distance = tempDistance
				bestIndex = i
			}
		}
		(bestIndex, centroids(bestIndex))
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

		val jobName = "KMedoids"

		val conf = new SparkConf().setAppName(jobName)
		val sc = new SparkContext(conf)
		val args = new Conf(argv)
		log.info("****** ~~~~~~ Input: " + args.input())
		log.info("****** ~~~~~~ Output: " + args.output())
		log.info("****** ~~~~~~ No. of Clusters: " + args.clusters())
		log.info("****** ~~~~~~ No. of iterations: " + args.iterations())
		FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)

		//		val hconf = new Configuration
		//		hconf.set("textinputformat.record.delimiter", "#Article:")

		//		val dataset = sc.newAPIHadoopFile(args.input(), classOf[ TextInputFormat ], classOf[ LongWritable ], classOf[ Text ], hconf)
		//			.map(x => x._2.toString())
		//			.filter(x => x.isEmpty() == false)
		//			.map(x => { val y = x.replaceAll("#Type: regular article", "")
		//						.replaceAll("[^a-zA-Z]", " ")
		//						.replaceAll("\\s\\s+", " ")
		//						.split(" ").toSeq.drop(1)
		//						
		//				(y.head,y)
		//			})

		val dataset = sc.textFile(args.input())
			.map(x => {
				val y = x.split("\\t")
				val z = y(0)
				val a = y(1).trim().split(" ").toSeq
				val b = a.filter(x => x.length() >= 4)
				(z, b)
			})

		val hashingTF = new HashingTF()

		val tf = dataset.map(x => (x._1, hashingTF.transform(x._2))).cache()
		val idf = new IDF(minDocFreq = 60).fit(tf.values)

		val tfidf = tf.map(x => (x._1, idf.transform(x._2)))

		val gg = tfidf.map(x => (x._1, x._2.toSparse))

		val articles = gg.map(x => (x._1, (x._2.indices zip x._2.values).toMap)).cache()

		//		println("Data: ")
		//		articles.foreach(println)

		var medoids = articles.takeSample(false, args.clusters().toInt).map(x => x._2)

		//		println("Start medoids")
		//		println(medoids.deep.mkString("\n"))

		var iteration = 1

		while (iteration <= args.iterations().toInt) {

			// Get the closest medoids to each article
			// and map them as medoids -> (article)
			val clusters = articles.map(article => (closestCentroid(article._2, medoids)._2, article._2))
								.combineByKey((value: Map[Int,Double]) => List(value),(aggr: List[Map[Int,Double]], value: Map[Int,Double]) => aggr ::: List(value),
											(aggr1: List[Map[Int,Double]], aggr2: List[Map[Int,Double]]) => aggr1 ::: aggr2,1)						

//			println("Clusters: ")
//			clusters.foreach(println)

			var test = clusters.map(f => {
				f._2.minBy(e => {
					f._2.foldLeft(0.0)((a, b) => a + cosineDistance(b, e))
				})
			}).coalesce(1, false).collect()

			var converge = (test zip medoids).map { case (a, b) => cosineDistance(a, b) }
//			println("Converge at " + iteration.toString() + " iteration")
//			println(converge.deep.mkString("\n"))

//			println("Converge length " + converge.length.toString() + " at " + iteration.toString() + " iteration")
			if (converge.filter(x => (x >= 0 && x < 0.000001)).length >= (args.clusters().toLong * 0.8)) {
				println("***** ~~~~~  Converged in " + iteration.toString() + " iterations")
				iteration = args.iterations().toInt;
			}
			
			medoids = test map (identity)

			//			println("Medoids at " + iteration.toString() + " iteration")
			//			println(medoids.deep.mkString("\n"))

			iteration = iteration + 1
		}

		val clusters = articles.map(article => (closestCentroid(article._2, medoids)._1, article._1))

		val numDocumentInClusters = clusters.groupByKey().map(f => (f._1, f._2.count(x => (x.isEmpty() == false))))

		println("***** ~~~~ Cluster -> No. of documents ")
		println(numDocumentInClusters.toArray().mkString("\n"))

		val numClusterForDocuments = clusters.map(f => (f._2, f._1)).groupByKey()

		numClusterForDocuments.coalesce(1, false).saveAsTextFile(args.output())
		//		val printThis = averageDistanceBetweenCentroids(medoids)
		//		println("***** ~~~~ Average Distance between Centroids: ")
		//		println(printThis.mkString("\n"))
	}
}