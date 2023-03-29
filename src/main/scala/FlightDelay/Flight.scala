package FlightDelay

import org.apache.log4j.{Level, Logger}
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StringIndexer, VectorAssembler, VectorIndexer}
import org.apache.spark.ml.regression.{GBTRegressor, LinearRegression, RandomForestRegressor}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col


object Flight {
  def main(args: Array[String]): Unit = {
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    Logger.getLogger("org.apache.spark").setLevel(Level.WARN)
    Logger.getLogger("org.spark-project").setLevel(Level.WARN)

    print("\n")
    print("Big Data Spark-Scala project.\n")
    print("Predicting the arrival Delay of Commercial Flights.\n")

    val dataPath = "hdfs://localhost:9000/data/data/2008.csv.bz2"
    var mlTechnique: Int = 0
    val spark = SparkSession.builder()
      .master("local[*]")
      .appName("Flight Delay")
      .getOrCreate()

    val rawData = spark.read.option("delimiter", ",")
      .option("header", "true")
      .csv(dataPath)
      .withColumn("DelayOutputVar", col("ArrDelay").cast("double"))
      .withColumn("DepDelayDouble", col("DepDelay").cast("double"))
      .withColumn("TaxiOutDouble", col("TaxiOut").cast("double"))
      .cache()

    //rawData.show()
    val data2 = rawData
      .drop("ActualElapsedTime") //forbidden
      .drop("ArrTime") //forbidden
      .drop("AirTime") //forbidden
      .drop("TaxiIn") //forbidden
      .drop("Diverted") //forbidden
      .drop("CarrierDelay") //forbidden
      .drop("WeatherDelay") //forbidden
      .drop("NASDelay") // Forbidden
      .drop("SecurityDelay") // Forbidden
      .drop("LateAircraftDelay") // Forbidden
      .drop("DepDelay") // Casted to double in a new variable called DepDelayDouble
      .drop("TaxiOut") // Casted to double in a new variable called TaxiOutDouble
      .drop("UniqueCarrier") // Always the same value // Remove correlated variables
      .drop("CancellationCode") // Cancelled flights don't count
      .drop("DepTime") // Highly correlated to CRSDeptime
      .drop("CRSArrTime") // Highly correlated to CRSDeptime
      .drop("CRSElapsedTime") // Highly correlated to Distance
      .drop("Distance") // Remove uncorrelated variables to the arrDelay
      .drop("FlightNum") // Remove uncorrelated variables to the arrDelay
      .drop("CRSDepTime") // Remove uncorrelated variables to the arrDelay
      .drop("Year") // Remove uncorrelated variables to the arrDelay
      .drop("Month") // Remove uncorrelated variables to the arrDelay
      .drop("DayofMonth") // Remove uncorrelated variables to the arrDelay
      .drop("DayOfWeek") // Remove uncorrelated variables to the arrDelay
      .drop("TailNum")

    //data2.show()
    //Remove cancelled flights
    val data3 = data2.filter("DelayOutputVar is not null")
    //data3.show()

    val originIndex = new StringIndexer()
      .setInputCol("Origin")
      .setOutputCol("OriginIndex")
      .fit(data3)
    val OriginIndexed = originIndex.transform(data3)
    //OriginIndexed.show()

    val OriginEncoder = new OneHotEncoder()
      .setInputCol("OriginIndex")
      .setOutputCol("OriginVec")
      .fit(OriginIndexed)
    val OriginEncoded = OriginEncoder.transform(OriginIndexed)
    //OriginEncoded.show()

    val DestIndex = new StringIndexer()
      .setInputCol("Dest")
      .setOutputCol("DestIndex")
      .fit(OriginEncoded)
    val DestIndexed = DestIndex.transform(OriginEncoded)
    //DestIndexed.show()

    val DestEncoder = new OneHotEncoder()
      .setInputCol("DestIndex")
      .setOutputCol("DestVec")
      .fit(DestIndexed)
    val data = DestEncoder.transform(DestIndexed)
    //data.show()

//    //**************  Linear Regression  ****************
//
//    val featureCols = Array("DepDelayDouble", "TaxiOutDouble", "OriginVec", "DestVec")
//    val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
//    val df = assembler.transform(data)
//    //df.show()
//
//    val Array(train, test) = df.randomSplit(Array(0.7, 0.3), seed=12345)
//    //train.show()
//
//    val lr = new LinearRegression()
//      .setLabelCol("DelayOutputVar")
//      .setFeaturesCol("features")
//      .setMaxIter(10)
//      .setRegParam(0.3)
//      .setElasticNetParam(0.8)
//
//    // Fit the model
//    val lrModel = lr.fit(train)
//    // Print the coefficients and intercept for linear regression
//    println(s"Coefficients: ${lrModel.coefficients} Intercept: ${lrModel.intercept}")
//
//    // Summarize the model over the training set and print out some metrics
//    val trainingSummary = lrModel.summary
//    println(s"numIterations: ${trainingSummary.totalIterations}")
//    println(s"objectiveHistory: [${trainingSummary.objectiveHistory.mkString(",")}]")
//    trainingSummary.residuals.show()
//    println(s"RMSE: ${trainingSummary.rootMeanSquaredError}")
//    println(s"r2: ${trainingSummary.r2}")
//
//    //Finally, Score
//
//    val predictions = lrModel.transform(test)
//    val evaluator = new RegressionEvaluator()
//      .setLabelCol("DelayOutputVar")
//      .setPredictionCol("prediction")
//      .setMetricName("r2")
//
//    val r2_linear = evaluator.evaluate(predictions)
//
//    println("R-sqr on test data = " + r2_linear)

    //    *********************    Ramdom forest Regression   **************************

        val featureCols = Array("DepDelayDouble", "TaxiOutDouble", "OriginIndex", "DestIndex")
        val assembler = new VectorAssembler().setInputCols(featureCols).setOutputCol("features")
        val df = assembler.transform(data)
        //df.show()

//        // Automatically identify categorical features, and index them.
//        // Set maxCategories so features with > 4 distinct values are treated as continuous.
//        val featureIndexer = new VectorIndexer()
//          .setInputCol("features")
//          .setOutputCol("indexedFeatures")
//          .setMaxCategories(291)
//          .fit(df)
//
//        // Split the data into training and test sets (30% held out for testing).
//        val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))
//
//        // Train a RandomForest model.
//        val rf = new RandomForestRegressor()
//          .setMaxBins(291)
//          .setMaxDepth(10)
//          .setLabelCol("DelayOutputVar")
//          .setFeaturesCol("indexedFeatures")
//
//        // Chain indexer and forest in a Pipeline.
//        val pipeline = new Pipeline()
//          .setStages(Array(featureIndexer, rf))
//
//        // Train model. This also runs the indexer.
//        val model = pipeline.fit(trainingData)
//
//        // Make predictions.
//        val predictions_ramdom = model.transform(testData)
//
//        // Select example rows to display.
//        predictions_ramdom.select("prediction", "DelayOutputVar", "features").show(5)
//
//        // Select (prediction, true label) and compute test error.
//        val evaluator_rf = new RegressionEvaluator()
//          .setLabelCol("DelayOutputVar")
//          .setPredictionCol("prediction")
//          .setMetricName("rmse")
//        val rmse = evaluator_rf.evaluate(predictions_ramdom)
//        println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

    /*    // val distinctValuesDF = df.select(df("age")).distinct
        val df1 = df.select(df("Dest")).distinct()
        df1.show()
        print("Total = " + df1.count())
        */
    // ****************  Gradient-boosted tree regression  ******************************

    // Automatically identify categorical features, and index them.
    // Set maxCategories so features with > 4 distinct values are treated as continuous.
    val featureIndexer = new VectorIndexer()
      .setInputCol("features")
      .setOutputCol("indexedFeatures")
      .setMaxCategories(291)
      .fit(df)

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = df.randomSplit(Array(0.7, 0.3))

    // Train a GBT model.
    val gbt = new GBTRegressor()
      .setLabelCol("DelayOutputVar")
      .setFeaturesCol("indexedFeatures")
      .setMaxIter(10)
      .setMaxBins(291)

    // Chain indexer and GBT in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(featureIndexer, gbt))

    // Train model. This also runs the indexer.
    val model = pipeline.fit(trainingData)

    // Make predictions.
    val predictions_gbt = model.transform(testData)

    // Select example rows to display.
    predictions_gbt.select("prediction", "DelayOutputVar", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator_gbt = new RegressionEvaluator()
      .setLabelCol("DelayOutputVar")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator_gbt.evaluate(predictions_gbt)
    println(s"Root Mean Squared Error (RMSE) on test data = $rmse")



  }

}
