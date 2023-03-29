ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.10"

lazy val root = (project in file("."))
  .settings(
    name := "CommercialFlight",
    libraryDependencies += "org.apache.spark" %% "spark-core" % "3.0.0",
    libraryDependencies += "org.apache.spark" %% "spark-sql" % "3.0.0",
    libraryDependencies += "org.apache.spark" %% "spark-mllib" % "3.0.0",
    libraryDependencies += "org.apache.spark" %% "spark-streaming" % "3.0.0",
    libraryDependencies += "org.apache.logging.log4j" % "log4j-api" % "2.20.0",
    libraryDependencies += "org.apache.logging.log4j" % "log4j-core" % "2.20.0"
  )
