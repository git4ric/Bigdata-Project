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
	    FileSystem.get(sc.hadoopConfiguration).delete(new Path(args.output()), true)
	    
	    val hconf = new Configuration
	    hconf.set("textinputformat.record.delimiter", "#Article:")
	    
	    val dataset = sc.newAPIHadoopFile(args.input(), classOf[TextInputFormat], classOf[LongWritable], classOf[Text], hconf)
	    				.map(x => x._2.toString().split(" ").toSeq)
	    				
	    val hashingTF = new HashingTF()
		val tf: RDD[Vector] = hashingTF.transform(dataset)
	    tf.cache()
		val idf = new IDF().fit(tf)
		val tfidf: RDD[Vector] = idf.transform(tf)
	}
}