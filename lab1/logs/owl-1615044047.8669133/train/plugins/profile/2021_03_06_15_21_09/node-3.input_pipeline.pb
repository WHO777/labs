  *???q_?AF??M?.KA2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2?l?????@!???b?I@)l?????@1???b?I@:Preprocessing2|
EIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch?n,(???@!?|&??B@)?n,(???@1?|&??B@:Preprocessing2?
?Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[3]::TFRecord?ʣa?J?@!?3o??)@)ʣa?J?@1?3o??)@:Advanced file read2?
cIterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl?xG?j;?@!G?S?I@)??߆@1>X?6???:Preprocessing2?
{Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap??5?;U?@!:h??ڗ)@)ǻ#c?9??1??hX????:Preprocessing2?
_Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle::Prefetch::MapAndBatch::MemoryCache?n3??@!1p?:?I@)?e??tg??1?X?M?	??:Preprocessing2r
;Iterator::Model::MaxIntraOpParallelism::FiniteTake::Shuffle{??ޝ?@!???h
?B@)a??+e??1R??DBK?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?I??靗@!????B@)??a?7?w?1?+V??"?:Preprocessing2F
Iterator::Modeln½2@!??[$?B@)?ڧ?1u?1?,[?v ?:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake?R#?㝗@!_??T?B@)[&??|t?1H?ĵ`a?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.