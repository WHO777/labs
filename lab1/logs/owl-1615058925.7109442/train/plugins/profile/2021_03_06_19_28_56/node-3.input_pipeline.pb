  *?v??b??@Zd{??-A2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2??`?????@!ߤ?h>R@)?`?????@1ߤ?h>R@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch?
?H??j@!t??˳4@)?
?H??j@1t??˳4@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[1]::TFRecord?a?4?AO@!r;??E@)a?4?AO@1r;??E@:Advanced file read2?
cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl?EGr??܇@!?E9{?VR@)?[?tYl@1?f??%??:Preprocessing2?
{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap??o??O@!軎?	Y@)U???N@??1??v???:Preprocessing2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache???`ľ??@!?TP@ZR@)?혺+;??1?	y????:Preprocessing2r
;Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle?R?Z?j@!?U-?4@)??Iط??1??&EN^?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism??fԭj@!8??4@)???@?w??1`?@??aL?:Preprocessing2F
Iterator::Model/3l?j@!??A??4@)7?X?O??1??yY?H?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake*??j@!??^KR?4@)???U?@x?1LH\?(?B?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.