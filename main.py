from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, round

# Initialize Spark session
spark = SparkSession.builder.appName("PySpark Data Analysis Homework").getOrCreate()

try:
    # Load the data
    users = spark.read.csv("users.csv", header=True, inferSchema=True)
    purchases = spark.read.csv("purchases.csv", header=True, inferSchema=True)
    products = spark.read.csv("products.csv", header=True, inferSchema=True)

    # Clean the data
    users = users.dropna()
    purchases = purchases.dropna()
    products = products.dropna()

    # Join the datasets
    joined_data = purchases \
        .join(users, "user_id") \
        .join(products, "product_id")

    # Total purchases by category
    total_by_category = joined_data \
        .withColumn("total_price", col("quantity") * col("price")) \
        .groupBy("category") \
        .agg(_sum("total_price").alias("total_spent")) \
        .orderBy(col("total_spent").desc())

    # Total purchases by category for age 18-25
    age_filtered_data = joined_data.filter(
        (col("age").isNotNull()) & (col("age") >= 18) & (col("age") <= 25)
    )
    total_by_category_age_18_25 = age_filtered_data \
        .withColumn("total_price", col("quantity") * col("price")) \
        .groupBy("category") \
        .agg(_sum("total_price").alias("total_spent")) \
        .orderBy(col("total_spent").desc())

    # Share of purchases for age 18-25
    total_spent_age_18_25 = total_by_category_age_18_25 \
        .agg(_sum("total_spent").alias("total_spent")).collect()[0]["total_spent"]

    percentage_by_category_age_18_25 = total_by_category_age_18_25 \
        .withColumn("percentage", round((col("total_spent") / total_spent_age_18_25) * 100, 2)) \
        .orderBy(col("percentage").desc())

    # Top 3 categories by percentage for age 18-25
    top_3_categories = percentage_by_category_age_18_25.limit(3)

    # Show results
    print("Total Purchases by Category:")
    total_by_category.show()

    print("Total Purchases by Category (Age 18-25):")
    total_by_category_age_18_25.show()

    print("Percentage by Category (Age 18-25):")
    percentage_by_category_age_18_25.show()

    print("Top 3 Categories by Percentage (Age 18-25):")
    top_3_categories.show()

finally:
    # Stop the Spark session
    spark.stop()
    print("Spark session stopped.")