  *?Mb?)?@?C?l?d?@)      ?=2?
XIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2@/?Xni`@!??^?GW@)/?Xni`@1??^?GW@:Preprocessing2?
nIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap[0]::TFRecord;??9y??"@!y?o?@)??9y??"@1y?o?@:Advanced file read2?
aIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::ParallelMapV2::FlatMap;?&?5?"@!?Ő??P@)?|?5^???1?N<#>???:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch?m?8)??!?f-Rl??)?m?8)??1?f-Rl??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatchJ?%r???!?џ?a??)J?%r???1?џ?a??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake?[w?T???!uj"f?Ɲ?)??ʉ?1?"wq????:Preprocessing2F
Iterator::Model???`???!j?fH???)??G?ֆ?1~?z?L???:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?Zd;??!?i?>5٠?)8?*5{?u?1?Ds?h^o?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.