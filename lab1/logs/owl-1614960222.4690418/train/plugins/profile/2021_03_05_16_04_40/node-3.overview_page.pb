?  *X94?	%AQ??+?,OA)       >2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2??˛Íѯ@!??zIRU@)?˛Íѯ@1??zIRU@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[4]::TFRecord??I(}ኅ@!ձR???,@)?I(}ኅ@1ձR???,@:Advanced file read2?
cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl?"4??{??@!?%?:?^U@)Eh???"@1?4D^??:Preprocessing2?
{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap??? v??@!Ni_???,@)?????$@1?x????:Preprocessing2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache????0 ??@!?^ӣ?`U@)]??%??1?a???V??:Preprocessing2r
;Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffled${??!??!??????}?)??`??1g9??z,t?:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch?e?O7P??!	/,??b?)?e?O7P??1	/,??b?:Preprocessing2?
RIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch?
??捓?!e???4:?)?
??捓?1e???4:?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelismz?I|???!?f?r??~?)?:pΈ?~?1??f?$?:Preprocessing2F
Iterator::Model	?f?ba??!?j$??U?)J}Yک?|?13?`??#?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake(??&2s??!??aBR~?)1%??et?1??MkVV?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.Y      Y@qK??Ds?,?"?
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