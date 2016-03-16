package com.examples

import org.apache.spark.SparkContext
import org.apache.spark.SparkContext._
import org.apache.spark.SparkConf
import org.apache.log4j.Logger

object KMeans {

  def main(arg: Array[String]) {

    var logger = Logger.getLogger(this.getClass())

    val jobName = "KMeans"

    val conf = new SparkConf().setAppName(jobName)
    val sc = new SparkContext(conf)

  }
}
