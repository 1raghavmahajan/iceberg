/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.iceberg;

import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.function.Supplier;
import java.util.stream.Collectors;
import java.util.stream.IntStream;
import org.apache.iceberg.expressions.Expression;
import org.apache.iceberg.io.CloseableIterable;
import org.apache.iceberg.io.ClosingIterator;
import org.apache.iceberg.io.FileIO;
import org.apache.iceberg.metrics.MetricsReporter;
import org.apache.iceberg.relocated.com.google.common.base.Joiner;
import org.apache.iceberg.relocated.com.google.common.collect.Iterables;
import org.apache.iceberg.relocated.com.google.common.collect.Lists;
import org.apache.iceberg.spark.JobGroupInfo;
import org.apache.iceberg.spark.JobGroupUtils;
import org.apache.iceberg.spark.SparkReadConf;
import org.apache.iceberg.spark.actions.ManifestFileBean;
import org.apache.iceberg.spark.source.SerializableTableWithSize;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.sql.SparkSession;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * A batch data scan that can utilize Spark cluster resources for planning.
 *
 * <p>This scan remotely filters manifests, fetching only the relevant data and delete files to the
 * driver. The delete file assignment is done locally after the remote filtering step. Such approach
 * is beneficial if the remote parallelism is much higher than the number of driver cores.
 *
 * <p>This scan is best suited for queries with selective filters on lower/upper bounds across all
 * partitions, or against poorly clustered metadata. This allows job planning to benefit from highly
 * concurrent remote filtering while not incurring high serialization and data transfer costs. This
 * class is also useful for full table scans over large tables but the cost of bringing data and
 * delete file details to the driver may become noticeable. Make sure to follow the performance tips
 * below in such cases.
 *
 * <p>Ensure the filtered metadata size doesn't exceed the driver's max result size. For large table
 * scans, consider increasing `spark.driver.maxResultSize` to avoid job failures.
 *
 * <p>Performance tips:
 *
 * <ul>
 *   <li>Enable Kryo serialization (`spark.serializer`)
 *   <li>Increase the number of driver cores (`spark.driver.cores`)
 *   <li>Tune the number of threads used to fetch task results (`spark.resultGetter.threads`)
 * </ul>
 */
public class SparkDistributedDataScan extends BaseDistributedDataScan {

  private static final Logger LOG = LoggerFactory.getLogger(SparkDistributedDataScan.class);

  private static final Joiner COMMA = Joiner.on(',');
  private static final String DELETE_PLANNING_JOB_GROUP_ID = "DELETE-PLANNING";
  private static final String DATA_PLANNING_JOB_GROUP_ID = "DATA-PLANNING";

  private final SparkSession spark;
  private final JavaSparkContext sparkContext;
  private final SparkReadConf readConf;

  private Broadcast<Table> tableBroadcast = null;
  private Integer limit = null;

  public SparkDistributedDataScan(SparkSession spark, Table table, SparkReadConf readConf) {
    this(spark, table, readConf, table.schema(), newTableScanContext(table));
  }

  protected SparkDistributedDataScan(
      SparkSession spark,
      Table table,
      SparkReadConf readConf,
      Schema schema,
      TableScanContext context) {
    super(table, schema, context);
    this.spark = spark;
    this.sparkContext = JavaSparkContext.fromSparkContext(spark.sparkContext());
    this.readConf = readConf;
  }

  @Override
  protected BatchScan newRefinedScan(
      Table newTable, Schema newSchema, TableScanContext newContext) {
    return new SparkDistributedDataScan(spark, newTable, readConf, newSchema, newContext);
  }

  @Override
  protected int remoteParallelism() {
    return readConf.parallelism();
  }

  @Override
  protected PlanningMode dataPlanningMode() {
    return readConf.dataPlanningMode();
  }

  @Override
  protected boolean shouldCopyRemotelyPlannedDataFiles() {
    return false;
  }

  @Override
  public BatchScan limit(Integer limit) {
    this.limit = limit;
    return this;
  }

  @Override
  protected Iterable<CloseableIterable<DataFile>> planDataRemotely(
      List<ManifestFile> dataManifests, boolean withColumnStats) {
    JobGroupInfo info = new JobGroupInfo(DATA_PLANNING_JOB_GROUP_ID, jobDesc("data"), true);
    return withJobGroupInfo(info, () -> doPlanDataRemotely(dataManifests, withColumnStats));
  }

  private Iterable<CloseableIterable<DataFile>> doPlanDataRemotely(
      List<ManifestFile> dataManifests, boolean withColumnStats) {
    scanMetrics().scannedDataManifests().increment(dataManifests.size());
    
    // If we have a limit, use the take approach to process manifests incrementally
    if (limit != null && !dataManifests.isEmpty()) {
      LOG.info("Using take-like approach for processing manifests with row limit {}", limit);
      return takeDataFromManifests(dataManifests, limit, withColumnStats);
    }
    
    // Without a limit, process all manifests as before
    JavaRDD<DataFile> dataFileRDD =
        sparkContext
            .parallelize(toBeans(dataManifests), dataManifests.size())
            .flatMap(new ReadDataManifest(tableBroadcast(), context(), withColumnStats));
    List<List<DataFile>> dataFileGroups = collectPartitions(dataFileRDD);

    int matchingFilesCount = dataFileGroups.stream().mapToInt(List::size).sum();
    int skippedFilesCount = liveFilesCount(dataManifests) - matchingFilesCount;
    scanMetrics().skippedDataFiles().increment(skippedFilesCount);

    return Iterables.transform(dataFileGroups, CloseableIterable::withNoopClose);
  }

  @Override
  protected PlanningMode deletePlanningMode() {
    return readConf.deletePlanningMode();
  }

  @Override
  protected DeleteFileIndex planDeletesRemotely(List<ManifestFile> deleteManifests) {
    JobGroupInfo info = new JobGroupInfo(DELETE_PLANNING_JOB_GROUP_ID, jobDesc("deletes"));
    return withJobGroupInfo(info, () -> doPlanDeletesRemotely(deleteManifests));
  }

  private DeleteFileIndex doPlanDeletesRemotely(List<ManifestFile> deleteManifests) {
    scanMetrics().scannedDeleteManifests().increment(deleteManifests.size());

    List<DeleteFile> deleteFiles =
        sparkContext
            .parallelize(toBeans(deleteManifests), deleteManifests.size())
            .flatMap(new ReadDeleteManifest(tableBroadcast(), context()))
            .collect();

    int skippedFilesCount = liveFilesCount(deleteManifests) - deleteFiles.size();
    scanMetrics().skippedDeleteFiles().increment(skippedFilesCount);

    return DeleteFileIndex.builderFor(deleteFiles)
        .specsById(table().specs())
        .caseSensitive(isCaseSensitive())
        .scanMetrics(scanMetrics())
        .build();
  }

  private <T> T withJobGroupInfo(JobGroupInfo info, Supplier<T> supplier) {
    return JobGroupUtils.withJobGroupInfo(sparkContext, info, supplier);
  }

  /**
   * Finds the highest sequence number among delete manifests.
   * This is useful to determine which data manifests might be affected by deletes.
   *
   * @param manifests List of all manifests
   * @return The highest sequence number of any delete manifest, or 0 if none found
   */
  private long findHighestDeleteSequenceNumber(List<ManifestFile> manifests) {
    long highest = 0L;
    for (ManifestFile manifest : manifests) {
      if (manifest.content() == ManifestContent.DELETES) {
        Long seqNum = manifest.sequenceNumber();
        if (seqNum != null && seqNum > highest) {
          highest = seqNum;
        }
      }
    }
    return highest;
  }

  /**
   * Takes data from manifests incrementally using a Spark take()-like approach.
   *
   * <p>This method follows the same pattern as Spark RDD.take():
   * 1. Start with a small batch of manifests
   * 2. Process them to collect data files
   * 3. Estimate how many more manifests we need based on current results
   * 4. Continue processing in increasing batches until we have enough data
   *
   * @param allManifests All manifests that could be processed
   * @param rowLimit Target number of rows to collect
   * @param withStats Whether to include column stats
   * @return Iterable of data files from processed manifests
   */
  private Iterable<CloseableIterable<DataFile>> takeDataFromManifests(
      List<ManifestFile> allManifests, long rowLimit, boolean withStats) {
    // First, sort manifests by relevance (sequence number, etc.)
    List<ManifestFile> sortedManifests = sortManifestsByRelevance(allManifests);

    // Find highest delete sequence number for proper row estimation
    long highestDeleteSeqNum = findHighestDeleteSequenceNumber(sortedManifests);

    // Variables to track our progress
    List<List<DataFile>> collectedGroups = Lists.newArrayList();
    long estimatedRowsCollected = 0;
    int manifestsProcessed = 0;
    final int totalManifests = sortedManifests.size();
    int scaleUpFactor = 2; // Double each time, like Spark

    while (estimatedRowsCollected < rowLimit && manifestsProcessed < totalManifests) {
      // Calculate how many manifests to try in this batch
      int manifestsToProcess;

      if (manifestsProcessed == 0) {
        // Start with a small batch (2 manifests)
        manifestsToProcess = Math.min(2, totalManifests);
      } else {
        // If we didn't find any files in the previous batch, multiply by scale factor and retry
        if (collectedGroups.stream().mapToInt(List::size).sum() == 0) {
          manifestsToProcess = Math.min(manifestsProcessed * scaleUpFactor, totalManifests - manifestsProcessed);
        } else {
          // Otherwise, estimate how many more manifests we need based on what we've processed so far
          double rowsPerManifest = (double) estimatedRowsCollected / manifestsProcessed;
          long rowsNeeded = rowLimit - estimatedRowsCollected;

          // Calculate needed manifests and overestimate by 50%
          int neededManifests = (int) Math.ceil(1.5 * rowsNeeded / rowsPerManifest);
          manifestsToProcess = Math.min(neededManifests, totalManifests - manifestsProcessed);

          // Also cap at scale factor times what we've processed so far to avoid huge jumps
          manifestsToProcess = Math.min(manifestsToProcess, manifestsProcessed * scaleUpFactor);
        }
      }

      // Ensure we process at least one manifest
      manifestsToProcess = Math.max(1, manifestsToProcess);

      LOG.info("Processing batch of {} manifests ({} to {})", manifestsToProcess,
          manifestsProcessed, manifestsProcessed + manifestsToProcess - 1);

      // Extract the batch of manifests to process
      List<ManifestFile> manifestBatch = sortedManifests.subList(
          manifestsProcessed, Math.min(manifestsProcessed + manifestsToProcess, totalManifests));

      // Process this batch of manifests
      JavaRDD<DataFile> batchRDD =
          sparkContext
              .parallelize(toBeans(manifestBatch), manifestBatch.size())
              .flatMap(new ReadDataManifest(tableBroadcast(), context(), withStats));

      List<List<DataFile>> batchGroups = collectPartitions(batchRDD);
      collectedGroups.addAll(batchGroups);

      // Count files and update metrics
      int filesInBatch = batchGroups.stream().mapToInt(List::size).sum();

      // Estimate rows based on files collected and manifest sequence numbers
      long batchRowEstimate = 0;
      for (ManifestFile manifest : manifestBatch) {
        batchRowEstimate += estimateRowsInManifest(manifest, highestDeleteSeqNum);
      }
      estimatedRowsCollected += batchRowEstimate;

      LOG.info("Batch collected {} files with estimated {} rows", filesInBatch, batchRowEstimate);

      // Update how many manifests we've processed
      manifestsProcessed += manifestsToProcess;

      // Check if we've collected enough data
      if (estimatedRowsCollected >= rowLimit) {
        LOG.info("Collected enough data to satisfy row limit: {} estimated rows", estimatedRowsCollected);
        break;
      }
    }

    // Update metrics about skipped files
    int processedFileCount = collectedGroups.stream().mapToInt(List::size).sum();
    int totalFileCount = liveFilesCount(sortedManifests.subList(0, manifestsProcessed));
    int skippedFileCount = totalFileCount - processedFileCount;
    scanMetrics().skippedDataFiles().increment(skippedFileCount);

    // Cancel any remaining jobs
    if (manifestsProcessed < totalManifests) {
      LOG.info("Cancelling remaining data planning jobs after processing {} of {} manifests",
          manifestsProcessed, totalManifests);
      cancelJobGroup(DATA_PLANNING_JOB_GROUP_ID);
    }

    return Iterables.transform(collectedGroups, CloseableIterable::withNoopClose);
  }


  /**
   * Cancel any running jobs in a job group.
   *
   * @param jobGroupId The job group ID to cancel
   */
  private void cancelJobGroup(String jobGroupId) {
    LOG.info("Cancelling job group {}", jobGroupId);
    sparkContext.cancelJobGroup(jobGroupId);
  }

  /**
   * Sorts manifests by relevance for limit processing with special consideration for sequence numbers.
   *
   * <p>This method sorts manifests to process the most relevant ones first:
   * 1. Higher sequence numbers first (more recent data)
   * 2. Lower min sequence numbers last (older data)
   * 3. Data manifests before delete manifests to ensure we have data before processing deletes
   *
   * @param manifests List of manifests to sort
   * @return Sorted list of manifests with most relevant first
   */
  private List<ManifestFile> sortManifestsByRelevance(List<ManifestFile> manifests) {
    List<ManifestFile> sorted = Lists.newArrayList(manifests);
    sorted.sort((m1, m2) -> {
      // First sort by content type - data files first, then deletes
      // This ensures we process data files before their corresponding deletes
      if (m1.content() != m2.content()) {
        // DATA (0) comes before DELETES (1)
        return m1.content().ordinal() - m2.content().ordinal();
      }

      // Next compare by sequence number (higher/newer first)
      // A null sequence number is considered oldest (lowest priority)
      if (m1.sequenceNumber() != m2.sequenceNumber()) {
        return Long.compare(m2.sequenceNumber(), m1.sequenceNumber());
      }

      // Next by min sequence number (higher/newer first)
      // This helps to prioritize manifests with newer data
      if (m1.minSequenceNumber() != m2.minSequenceNumber()) {
        return Long.compare(m2.minSequenceNumber(), m1.minSequenceNumber());
      }

      // Then by snapshot ID (most recent first)
      int result = Long.compare(m2.snapshotId(), m1.snapshotId());
      if (result != 0) {
        return result;
      }

      // Finally by file counts (larger manifests first)
      // For data manifests, prioritize by added files; for delete manifests, by deleted files
      if (m1.content() == ManifestContent.DATA) {
        // For data manifests, prioritize by added files count
        return Integer.compare(
            nullToZero(m2.addedFilesCount()) + nullToZero(m2.existingFilesCount()),
            nullToZero(m1.addedFilesCount()) + nullToZero(m1.existingFilesCount()));
      } else {
        // For delete manifests, prioritize by deleted files count
        return Integer.compare(
            nullToZero(m2.deletedFilesCount()),
            nullToZero(m1.deletedFilesCount()));
      }
    });
    return sorted;
  }

  private static int nullToZero(Integer value) {
    return value != null ? value : 0;
  }

  /**
   * Estimates the number of rows in a manifest with special handling for delete manifests and
   * consideration of sequence numbers.
   *
   * <p>For delete manifests, estimates are less reliable. For data manifests, we need to consider
   * that data files coming after deletes cannot be safely estimated.
   *
   * @param manifest The manifest to estimate rows for
   * @param highestDeleteSeqNum The highest sequence number of any delete manifest seen
   * @return Estimated number of rows in the manifest
   */
  private long estimateRowsInManifest(ManifestFile manifest, long highestDeleteSeqNum) {
    // For data manifests, we need to consider sequence numbers relative to delete files
    Long seqNum = manifest.sequenceNumber();

    // If this data manifest has a sequence number LOWER than any delete manifest,
    // then we can use row counts as they are likely valid
    if (seqNum < highestDeleteSeqNum) {
      // Use available row counts
      long estimatedRows = 0;

      if (manifest.hasAddedFiles() && manifest.addedRowsCount() != null) {
        estimatedRows += manifest.addedRowsCount();
      }

      if (manifest.hasExistingFiles() && manifest.existingRowsCount() != null) {
        estimatedRows += manifest.existingRowsCount();
      }

      if (estimatedRows > 0) {
        return estimatedRows;
      }
    }

    // For data manifests with sequence number >= highestDeleteSeqNum or when row counts aren't available,
    // estimate based on file counts with higher estimate per file since we need to read them
    int addedFiles = nullToZero(manifest.addedFilesCount());
    int existingFiles = nullToZero(manifest.existingFilesCount());
    long fileCount = addedFiles + existingFiles;

    // Use a higher estimate for files that might be affected by deletes
    long rowsPerFile = seqNum >= highestDeleteSeqNum ? 150_000L : 100_000L;
    return fileCount * rowsPerFile;
  }

  private String jobDesc(String type) {
    List<String> options = Lists.newArrayList();
    options.add("snapshot_id=" + snapshot().snapshotId());
    String optionsAsString = COMMA.join(options);
    return String.format("Planning %s (%s) for %s", type, optionsAsString, table().name());
  }

  private List<ManifestFileBean> toBeans(List<ManifestFile> manifests) {
    return manifests.stream().map(ManifestFileBean::fromManifest).collect(Collectors.toList());
  }

  private Broadcast<Table> tableBroadcast() {
    if (tableBroadcast == null) {
      Table serializableTable = SerializableTableWithSize.copyOf(table());
      this.tableBroadcast = sparkContext.broadcast(serializableTable);
    }

    return tableBroadcast;
  }

  private <T> List<List<T>> collectPartitions(JavaRDD<T> rdd) {
    int[] partitionIds = IntStream.range(0, rdd.getNumPartitions()).toArray();
    return Arrays.asList(rdd.collectPartitions(partitionIds));
  }

  private int liveFilesCount(List<ManifestFile> manifests) {
    return manifests.stream().mapToInt(this::liveFilesCount).sum();
  }

  private int liveFilesCount(ManifestFile manifest) {
    return manifest.existingFilesCount() + manifest.addedFilesCount();
  }

  private static TableScanContext newTableScanContext(Table table) {
    if (table instanceof BaseTable) {
      MetricsReporter reporter = ((BaseTable) table).reporter();
      return ImmutableTableScanContext.builder().metricsReporter(reporter).build();
    } else {
      return TableScanContext.empty();
    }
  }

  private static class ReadDataManifest implements FlatMapFunction<ManifestFileBean, DataFile> {

    private final Broadcast<Table> table;
    private final Expression filter;
    private final boolean withStats;
    private final boolean isCaseSensitive;

    ReadDataManifest(Broadcast<Table> table, TableScanContext context, boolean withStats) {
      this.table = table;
      this.filter = context.rowFilter();
      this.withStats = withStats;
      this.isCaseSensitive = context.caseSensitive();
    }

    @Override
    public Iterator<DataFile> call(ManifestFileBean manifest) throws Exception {
      FileIO io = table.value().io();
      Map<Integer, PartitionSpec> specs = table.value().specs();
      return new ClosingIterator<>(
          ManifestFiles.read(manifest, io, specs)
              .select(withStats ? SCAN_WITH_STATS_COLUMNS : SCAN_COLUMNS)
              .filterRows(filter)
              .caseSensitive(isCaseSensitive)
              .iterator());
    }
  }

  private static class ReadDeleteManifest implements FlatMapFunction<ManifestFileBean, DeleteFile> {

    private final Broadcast<Table> table;
    private final Expression filter;
    private final boolean isCaseSensitive;

    ReadDeleteManifest(Broadcast<Table> table, TableScanContext context) {
      this.table = table;
      this.filter = context.rowFilter();
      this.isCaseSensitive = context.caseSensitive();
    }

    @Override
    public Iterator<DeleteFile> call(ManifestFileBean manifest) throws Exception {
      FileIO io = table.value().io();
      Map<Integer, PartitionSpec> specs = table.value().specs();
      return new ClosingIterator<>(
          ManifestFiles.readDeleteManifest(manifest, io, specs)
              .select(DELETE_SCAN_WITH_STATS_COLUMNS)
              .filterRows(filter)
              .caseSensitive(isCaseSensitive)
              .iterator());
    }
  }
}
