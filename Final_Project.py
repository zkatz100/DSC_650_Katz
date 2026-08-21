from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import FloatType
from pyspark.ml.features import StringIndexer, VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

import happybase

# start Spark
spark = SparkSession.builder.appName('Predict_Math').enableHiveSupport().getOrCreate()

# import the Hive table into Spark
student_data = spark.read.table('performance')

# convert all the numeric columns to floats
num_columns = ['age',
               'study_hours',
               'attendance_percentage',
               'science_score',
               'english_score'
               ]
target_column = 'math_score'

for column in num_columns:
    student_data = student_data.withColumn(column, col(column).cast(FloatType()))

student_data = student_data.withColumn(target_column, col(target_column).cast(FloatType()))

# Encode categorical columns into numbers
# First make a list of all the categorical columns
cat_columns = ['gender',
              'school_type',
              'parent_education',
              'internet_access',
              'travel_time',
              'extra_activities',
              'study_method'
              ]
# Convert the strings to a numeric index
indexer = [StringIndexer(inputCol = column, outputCol = f'{col}_index', handleInvalid = 'keep') for column in cat_columns]

# turn each row of encoded categorical columns and numeric columns into a vector.
encoded_columns = [f'{column}_index' for column in cat_columns]
assembler = VectorAssembler(inputCols = encoded_columns + num_columns, outputCol = 'features')


# define the decision tree
tree = DecisionTreeClassifier(featuresCol = 'features', labelCol = target_column)

# define a pipeline to run the indexers and the tree
pipeline = Pipeline(stages = indexer + [assembler, tree])

# split into training and test data.
train_df, test_df = student_data.randomSplit([0.8,0.2], seed = 19)

# train the pipeline
pipeline_model = pipeline.fit(train_df)
# test the model
predictions = pipeline_model.transform(test_df)


# calculate evaluation metrics for the model
accuracy = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'accuracy')
f1 = MulticlassClassificationEvaluator(labelCol = 'label', predictionCol = 'prediction', metricName = 'f1')


accuracy_value = accuracy.evaluate(predictions)
f1_value = f1.evaluate(predictions)

# Connect to HBase using Happybase
connection = happybase.Connection('hbase-thrift-host')
table = connection.table('Models')

table.put(1,{
    'model_type': 'decision_tree',
    'metrics:accuracy': str(accuracy_value),
    'metrics:f1_score': str(f1_value)
})

print("Metrics written to HBase.")

spark.stop()