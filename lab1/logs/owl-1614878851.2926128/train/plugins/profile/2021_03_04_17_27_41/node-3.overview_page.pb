?  *ffff???@#??~#A2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2@$}ZŠl@!ݖVsV@)$}ZŠl@1ݖVsV@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord@k+??݋8@!!?v??#@)k+??݋8@1!?v??#@:Advanced file read2?
cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl@'L?J?l@!??zT?V@)??M????1??;?\I??:Preprocessing2?
{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap@?4?Ry?8@!??ʶ^#@)Y?n????1???3T??:Preprocessing2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache@Ō?? ?l@!?vj?V@)Wya?X??1)?!-?W??:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::PrefetchC???|Ͳ?!?H?p=}??)C???|Ͳ?1?H?p=}??:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatchw?$$?6??!p?8u???)w?$$?6??1p?8u???:Preprocessing2r
;Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle?
?lw??!Tt.xN??)?drjg?z?1?T???d?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismL?
F%u??!U?0?V???)?fh<q?1??3?4[?:Preprocessing2F
Iterator::Modelm?s?p??!??8??a??)0?x??no?1C !?X?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake????^a??!?C?VĠ?)ʩ?ajKm?1<????V?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@q?L,??N?"?
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)Q
Otf_data_bottleneck_analysis (find the bottleneck in the tf.data input pipeline)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"O
Mtensorflow_stats (identify the time-consuming operations executed on the GPU)"U
Strace_viewer (look at the activities on the timeline of each GPU in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2M
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono2no:
Refer to the TF2 Profiler FAQ2"GPU(: B??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Anode-3: Failed to load libcupti (is it installed and accessible?)