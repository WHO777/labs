  *? ?r(Q?@??|??<?@2?
iIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2;?w??1?T@!?q*?gV@)?w??1?T@1?q*?gV@:Preprocessing2?
Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap[0]::TFRecord<? Pō?!@!?e{?8S#@)? Pō?!@1?e{?8S#@:Advanced file read2?
ZIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl;??ފĦT@!?#? ??V@)??s?????1s^ְ??:Preprocessing2?
rIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCacheImpl::ParallelMapV2::FlatMap<L?^I?!@!?YY?ގ#@) A?c?]??1?"???ҽ?:Preprocessing2?
VIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch::MemoryCache;? OZ?T@!???*??V@)??W??1??8c?X??:Preprocessing2?
IIterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch::MapAndBatch??Hi6???!?????P??)??Hi6???1?????P??:Preprocessing2s
<Iterator::Model::MaxIntraOpParallelism::FiniteTake::Prefetch7??͟?!6R? iT??)7??͟?16R? iT??:Preprocessing2i
2Iterator::Model::MaxIntraOpParallelism::FiniteTake$?w~Q???!?C?ˇ??) ????m??1???͌?:Preprocessing2F
Iterator::ModelR?hE??!E?9o?ή?)???\??x?1?{q0{?:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism{?\?&??!??M?h??)?/???"u?1?o7|?w?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.